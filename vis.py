from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotsoccer as mps
import numpy as np
import math


def dual_axes(figsize=4):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches((figsize * 3, figsize))
    return axs[0], axs[1]


def loc_angle_axes(figsize=4):
    fig, _axs = plt.subplots(1, 2)
    fig.set_size_inches((figsize * 3, figsize))

    axloc = plt.subplot(121)
    axloc = field(axloc)
    axpol = plt.subplot(122, projection="polar")
    # axpol.set_rticks(np.linspace(0, 2, 21))
    return axloc, axpol


def field(ax):
    ax = mps.field(ax=ax, show=False)
    ax.set_xlim(-1, 105 + 1)
    ax.set_ylim(-1, 68 + 1)
    return ax


def movement(ax):
    plt.axis("on")
    plt.axis("scaled")
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    return ax


def polar(ax):
    plt.axis("on")
    ax.set_xlim(-3.2, 3.2)
    ax.spines["left"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    return ax


##################################
# MODEL-BASED VISUALIZATION
#################################


def show_location_model(loc_model, show=True, figsize=6):
    ax = mps.field(show=False, figsize=figsize)

    norm_strengths = loc_model.priors / np.max(loc_model.priors) * 0.8
    for strength, gauss, color in zip(norm_strengths, loc_model.submodels, colors * 10):
        add_ellips(ax, gauss.mean, gauss.cov, color=color, alpha=strength)
    if show:
        plt.show()
        

def show_direction_model(gauss, dir_models, show=True, figsize=6):
    ax = mps.field(show=False, figsize=figsize)
    
#     for gauss in loc_model.submodels:
    add_ellips(ax, gauss.mean, gauss.cov, alpha=0.5)
    
    x, y = gauss.mean

    for vonmises in dir_models.submodels:
        dx = np.cos(vonmises.loc)[0]
        dy = np.sin(vonmises.loc)[0]
        r = vonmises.R[0]
        add_arrow(ax, x, y, 10*dx, 10*dy, 
            linewidth=0.5)
            
    if show:
        plt.show()
    


def show_location_models(loc_models, figsize=6):
    """
    Model-based visualization
    """
    for model in loc_models:
        print(model.name, model.n_components)
        show_location_model(model, figsize=6)
        


def show_all_models(loc_models, dir_models):
    
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        ax = mps.field(show=False, figsize=8)
        
        am_subclusters = []
        for a, _ in enumerate(loc_model.submodels):
            for dir_model in dir_models:
                if f"{loc_model.name}_{a}" == dir_model.name:
                    am_subclusters.append(dir_model.n_components)
        
        am_subclusters = np.array(am_subclusters)
                
        for i, gauss in enumerate(loc_model.submodels):
                      
            if (am_subclusters == 1).all():
                    add_ellips(ax, gauss.mean, gauss.cov, alpha=0.5)
            
            else:
                add_ellips(ax, gauss.mean, gauss.cov, color='grey')

                x, y = gauss.mean
                for dir_model in dir_models:
                    if f"{loc_model.name}_{i}" == dir_model.name:
                        print(dir_model.name, dir_model.n_components)

                        for j, vonmises in enumerate(dir_model.submodels):
                                dx = np.cos(vonmises.loc)[0]
                                dy = np.sin(vonmises.loc)[0]
                                r = vonmises.R[0]
                                add_arrow(ax, x, y, 10*dx, 10*dy, 
                                              linewidth=0.5)
                        
        plt.show()
        

def show_direction_models(loc_models, dir_models, figsize=8):
    """
    Model-based visualization
    """
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        ax = mps.field(show=False, figsize=figsize)

        norm_strengths = loc_model.priors / np.max(loc_model.priors) * 0.8
        for i, (strength, gauss) in enumerate(zip(norm_strengths, loc_model.submodels)):
            add_ellips(ax, gauss.mean, gauss.cov, alpha=strength)

            x, y = gauss.mean
            for dir_model in dir_models:
                if f"{loc_model.name}_{i}" == dir_model.name:
                    print(dir_model.name, dir_model.n_components)
                    dir_norm_strengths = (
                        dir_model.priors / np.max(dir_model.priors) * 0.8
                    )
                    for strength, vonmises in zip(
                        dir_norm_strengths, dir_model.submodels
                    ):
                        dx = np.cos(vonmises.loc)[0]
                        dy = np.sin(vonmises.loc)[0]
                        r = vonmises.R[0]
                        add_arrow(
                            ax,
                            x,
                            y,
                            10 * r * dx,
                            10 * r * dy,
                            alpha=strength,
                            threshold=0,
                        )
        plt.show()


def add_ellips(ax, mean, covar, color=None, alpha=0.7):
    v, w = linalg.eigh(covar)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
    # ell.set_clip_box(axs[0].bbox)
    ell.set_alpha(alpha)
    ell.width = max(ell.width, 3)
    ell.height = max(ell.height, 3)
    ax.add_artist(ell)
    return ax


def add_arrow(ax, x, y, dx, dy, arrowsize=2.5, linewidth=2, threshold=2, alpha=1, fc='black', ec='black'):
    if abs(dx) > threshold or abs(dy) > threshold:
        return ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=arrowsize,
            head_length=arrowsize,
            linewidth=linewidth,
            fc=fc,  # colors[i % len(colors)],
            ec=ec,  # colors[i % len(colors)],
            length_includes_head=True,
            alpha=alpha,
            zorder=3,
        )


######################################################
# PROBABILITY-DENSITY-FUNCTION BASED VISUALIZATION
######################################################


def show_direction_models_pdf(loc_models, dir_models):
    """
    Probability-density function based visualization
    """
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        for i, gauss in enumerate(loc_model.submodels):
            # axloc, axpol = dual_axes()
            # # vis.add_ellips(axloc,gauss.mean,gauss.cov)
            # draw_contour(axloc, gauss, cmap="Blues")
            for dir_model in dir_models:
                if f"{loc_model.name}_{i}" == dir_model.name:
                    print(dir_model.name, dir_model.n_components)

                    axcol, axpol = loc_angle_axes()
                    draw_contour(axcol, gauss, cmap="Blues")
                    draw_vonmises_pdfs(dir_model, axpol)
                    plt.show()


def draw_contour(ax, gauss, n=100, cmap="Blues"):
    x = np.linspace(0, 105, n)
    y = np.linspace(0, 105, n)
    xx, yy = np.meshgrid(x, y)
    zz = gauss.pdf(np.array([xx.flatten(), yy.flatten()]).T)
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=cmap)
    return ax


def draw_vonmises_pdfs(model, ax=None,figsize=4,projection="polar",n=200,show=True):
    if ax is None:
        ax = plt.subplot(111, projection=projection)
        plt.gcf().set_size_inches((figsize, figsize))
    x = np.linspace(-np.pi, np.pi, n)
    total = np.zeros(x.shape)
    for i, (prior, vonmises) in enumerate(zip(model.priors, model.submodels)):
        p = prior * vonmises.pdf(x)
        p = np.nan_to_num(p)
        ax.plot(x, p, linewidth=2, color=(colors * 10)[i],label = f"Component {i}")
        total += p
#     ax.plot(x, total, linewidth=3, color="black")
    return ax


#################################
# DATA-BASED VISUALIZATION
#################################

colors = [
    "#377eb8",
    "#e41a1c",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]


def scatter_location_model(
    loc_model, actions, W, samplefn="max", tol=0.1, figsize=6, alpha=0.5, show=True
):
    X = actions[["x", "y"]]
    probs = loc_model.predict_proba(X, W[loc_model.name].values)
    probs = np.nan_to_num(probs)
    pos_prob_idx = probs.sum(axis=1) > tol
    x = X[pos_prob_idx]
    w = probs[pos_prob_idx]

    if loc_model.n_components > len(colors):
        means = [m.mean for m in loc_model.submodels]
        good_colors = color_submodels(means, colors)
    else:
        good_colors = colors
    c = scattercolors(w, good_colors, samplefn=samplefn)

    ax = mps.field(show=False, figsize=figsize)
    ax.scatter(x.x, x.y, c=c, alpha=alpha)
    if show:
        plt.show()

def scatter_location_model_black(
    loc_model, actions, W, samplefn="max", tol=0.1, figsize=6, alpha=0.5, show=True
):
    X = actions[["x", "y"]]
    probs = loc_model.predict_proba(X, W[loc_model.name].values)
    probs = np.nan_to_num(probs)
    pos_prob_idx = probs.sum(axis=1) > tol
    x = X[pos_prob_idx]
    w = probs[pos_prob_idx]

    if loc_model.n_components > len(colors):
        means = [m.mean for m in loc_model.submodels]
        good_colors = color_submodels(means, colors)
    else:
        good_colors = colors
    c = scattercolors(w, good_colors, samplefn=samplefn)

    ax = mps.field(show=False, figsize=figsize)
    ax.scatter(x.x, x.y, c="black", alpha=alpha)
    if show:
        plt.show()
        

def scatter_location_models(
    loc_models, actions, W, samplefn="max", tol=0.1, figsize=8, alpha=0.5
):
    """
    Data-based visualization
    """
    for model in loc_models:
        print(model.name, model.n_components)
        X = actions[["x", "y"]]
        probs = model.predict_proba(X, W[model.name].values)
        probs = np.nan_to_num(probs)
        pos_prob_idx = probs.sum(axis=1) > tol
        x = X[pos_prob_idx]
        w = probs[pos_prob_idx]

        if model.n_components > len(colors):
            means = [m.mean for m in model.submodels]
            good_colors = color_submodels(means, colors)
        else:
            good_colors = colors
        c = scattercolors(w, good_colors, samplefn=samplefn)

        ax = mps.field(show=False, figsize=figsize)
        ax.scatter(x.x, x.y, c=c, alpha=alpha)
        plt.show()


def scatter_direction_models(
    dir_models, actions, X, W, samplefn="max", tol=0.1, figsize=4, alpha=0.5
):
    for model in dir_models:
        print(model.name, model.n_components)
        probs = model.predict_proba(X, W[model.name].values)
        probs = np.nan_to_num(probs)
        pos_prob_idx = probs.sum(axis=1) > tol
        w = probs[pos_prob_idx]
        c = scattercolors(w, samplefn=samplefn)

        axloc, axmov = dual_axes()
        field(axloc)
        movement(axmov)

        x = actions[pos_prob_idx]
        axloc.scatter(x.x, x.y, c=c, alpha=alpha)
        axmov.scatter(x.dx, x.dy, c=c, alpha=alpha)
        plt.show()


def hist_direction_model(
    dir_model,
    actions,
    W,
    samplefn="max",
    tol=0.1,
    figsize=4,
    alpha=0.5,
    projection="polar",
    bins=20,
    show=False,
):
    X = actions["mov_angle_a0"]
    probs = dir_model.predict_proba(X, W[dir_model.name].values)
    probs = np.nan_to_num(probs)
    pos_prob_idx = probs.sum(axis=1) > tol
    w = probs[pos_prob_idx]
    c = scattercolors(w, samplefn=samplefn)

    axpol = plt.subplot(111, projection=projection)
    plt.gcf().set_size_inches((figsize, figsize))

    x = actions[pos_prob_idx]
    for p, c in zip(w.T, colors):
        p = p.flatten()
        axpol.hist(x.mov_angle_a0, weights=p.flatten(), color=c, alpha=alpha, bins=bins)
    if show:
        plt.show()




def hist_direction_models(
    dir_models, actions, W, samplefn="max", tol=0.1, figsize=4, alpha=0.5
):
    for model in dir_models:
        print(model.name, model.n_components)
        X = actions["mov_angle_a0"]
        probs = model.predict_proba(X, W[model.name].values)
        probs = np.nan_to_num(probs)
        pos_prob_idx = probs.sum(axis=1) > tol
        w = probs[pos_prob_idx]
        c = scattercolors(w, samplefn=samplefn)

        # axloc, axmov = dual_axes()
        # field(axloc)
        # movement(axmov)
        axloc, axpol = loc_angle_axes()

        x = actions[pos_prob_idx]
        axloc.scatter(x.x, x.y, c=c, alpha=alpha)
        for p, c in zip(w.T, colors):
            p = p.flatten()
            axpol.hist(
                x.mov_angle_a0, weights=p.flatten(), color=c, alpha=alpha, bins=100
            )
        # axpol.hist(x.mov_angle_a0, c=c, alpha=alpha)
        plt.show()


def model_vs_data(
    dir_models, loc_models, actions, W, samplefn="max", tol=0.1, figsize=4, alpha=0.5
):
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        for i, gauss in enumerate(loc_model.submodels):
            # axloc, axpol = dual_axes()
            # # vis.add_ellips(axloc,gauss.mean,gauss.cov)
            # draw_contour(axloc, gauss, cmap="Blues")
            for dir_model in dir_models:
                if f"{loc_model.name}_{i}" == dir_model.name:

                    print(dir_model.name, dir_model.n_components)
                    axcol, axpol = loc_angle_axes()
                    draw_contour(axcol, gauss, cmap="Blues")
                    draw_vonmises_pdfs(axpol, dir_model)
                    plt.show()

                    X = actions["mov_angle_a0"]
                    probs = dir_model.predict_proba(X, W[dir_model.name].values)
                    probs = np.nan_to_num(probs)
                    pos_prob_idx = probs.sum(axis=1) > tol
                    w = probs[pos_prob_idx]
                    c = scattercolors(w, samplefn=samplefn)

                    # axloc, axmov = dual_axes()
                    # field(axloc)
                    # movement(axmov)
                    axloc, axpol = loc_angle_axes()

                    x = actions[pos_prob_idx]
                    axloc.scatter(x.x, x.y, c=c, alpha=alpha)
                    for p, c in zip(w.T, colors):
                        p = p.flatten()
                        axpol.hist(
                            x.mov_angle_a0,
                            weights=p.flatten(),
                            color=c,
                            alpha=alpha,
                            bins=100,
                        )
                    # axpol.hist(x.mov_angle_a0, c=c, alpha=alpha)
                    plt.show()


from scipy.spatial import Delaunay
import networkx as nx


def color_submodels(means, colors):
    tri = Delaunay(means)
    edges = set()
    for s in tri.simplices:
        [a, b, c] = s
        es = set([frozenset([a, b]), frozenset([b, c]), frozenset([c, a])])
        edges = edges | es
    G = nx.Graph()
    for e in edges:
        [i, j] = list(e)
        G.add_edge(i, j)

    if len(G.nodes) > 0:
        r_ = max([G.degree(node) for node in G.nodes])
    else:
        r_ = 0
    if r_ > len(colors) - 1:
        colorassign = nx.algorithms.coloring.greedy_color(G)
    else:
        colorassign = nx.algorithms.coloring.equitable_color(G, len(colors))
    colorvector = [0] * len(means)
    for k, v in colorassign.items():
        colorvector[k] = int(v)

    return [colors[i] for i in colorvector]


def sample(probs):
    return np.random.choice(len(probs), p=probs / sum(probs))


def scattercolors(weights, colors=colors, samplefn="max"):
    if samplefn == "max":
        labels = np.argmax(weights, axis=1)
    else:
        labels = np.apply_along_axis(sample, axis=1, arr=weights)

    pcolors = [colors[l % len(colors)] for l in labels]
    return pcolors


#################################
# EXPERIMENTS VISUALIZATION
#################################


def savefigure(figname):
    plt.savefig(figname,dpi=300,
               bbox_inches="tight",
               pad_inches=0.0
               )
    
    
def show_component_differences(loc_models, dir_models, vec_p1, vec_p2, name1, name2, save=True):
    
    # determine colors of dir sub models
    difference = vec_p1 - vec_p2
    cmap = mpl.cm.get_cmap('bwr_r')
    
    for loc_model in loc_models:
        
        mini = min(difference.loc[difference.index.str.contains(f"^{loc_model.name}_")])
        maxi = max(difference.loc[difference.index.str.contains(f"^{loc_model.name}_")])
        ab = max(abs(mini), abs(maxi))
        
        if (ab == 0):
            ab = 0.0001
        
        norm = mpl.colors.DivergingNorm(vcenter=0, vmin=-ab,
                                               vmax = ab)
        
        print(loc_model.name, loc_model.n_components)
        ax = mps.field(show=False, figsize=8)
        
        
        am_subclusters = []
        for a, _ in enumerate(loc_model.submodels):
            for dir_model in dir_models:
                if f"{loc_model.name}_{a}" == dir_model.name:
                    am_subclusters.append(dir_model.n_components)
        
        am_subclusters = np.array(am_subclusters)
                
        for i, gauss in enumerate(loc_model.submodels):
                      
            if (am_subclusters == 1).all():
                    add_ellips(ax, gauss.mean, gauss.cov, 
                                   color=cmap(norm(difference.loc[f"{loc_model.name}_{i}_0"])), alpha=1)
            
            else:
                add_ellips(ax, gauss.mean, gauss.cov, color='gainsboro')

                x, y = gauss.mean
                for dir_model in dir_models:
                    if f"{loc_model.name}_{i}" == dir_model.name:
                        print(dir_model.name, dir_model.n_components)

                        for j, vonmises in enumerate(dir_model.submodels):
                                dx = np.cos(vonmises.loc)[0]
                                dy = np.sin(vonmises.loc)[0]
                                add_arrow(ax, x, y, 10*dx, 10*dy, 
                                              fc=cmap(norm(difference.loc[f"{loc_model.name}_{i}_{j}"])), 
                                              arrowsize=4.5, linewidth=1
                                             )
                        
        cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.065, pad=-0.05, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(labelsize=16) 
        plt.axis("scaled")
        
        if save:
            savefigure(f"../figures/{name1}-{name2}-{loc_model.name}.png")
        else:
            plt.show()
