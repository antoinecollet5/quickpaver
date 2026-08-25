import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Paving (tiling) examples

    In this tutorial, we will see how to generate tiling using rectangles, triangles
    and hexagons. We will also play with anisotropy and angles.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Start by importing the required modules
    """)
    return


@app.cell
def _():
    from typing import Optional, Union

    import matplotlib.pyplot as plt
    import nested_grid_plotter as ngp
    import quickpaver
    import shapely
    from shapely.plotting import plot_polygon

    return Optional, Union, ngp, plot_polygon, plt, quickpaver, shapely


@app.cell
def _(mo):
    mo.md(r"""
    There is a choice between three shapes: **triangle**, **rectangle**, and
    **hexagon**. By default, the shapes are regular: the triangle is **equilateral**,
    the rectangle is a **square**, and the hexagon is not distorted. The **anisotropy**
    allows the shape to be stretched or shortened along the y-axis.
    """)
    return


@app.cell
def _(ngp, plot_polygon, plt, quickpaver):
    def make_polygons():
        plotter = ngp.Plotter(
            plt.figure(figsize=(8, 8), constrained_layout=True),
            builder=ngp.SubplotsMosaicBuilder(
                mosaic=[[f"ax{i}-{j}" for j in range(3)] for i in range(3)],
                sharex=True,
                sharey=True,
            ),
        )

        for i, poly_type in enumerate(quickpaver.PolygonType.to_list()):
            for j, anisotropy_ratio in enumerate([1.0, 2.0, 0.5]):
                ax = plotter.ax_dict[f"ax{i}-{j}"]
                plot_polygon(
                    quickpaver.gen_polygon(
                        poly_type.value,
                        edge_length=30.0,
                        anisotropy_ratio=anisotropy_ratio,
                    ),
                    ax=ax,
                )
                ngp.hide_axis_spine(ax, loc="all")
                ax.set_aspect("equal")
                ngp.hide_axis_ticklabels(ax)
                ax.set_title(f"Anisotropy\nratio = {anisotropy_ratio:.1f}")
        return plotter.fig

    return (make_polygons,)


@app.cell
def _(make_polygons):
    make_polygons()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Now let’s take an example and load a simplified outline of France and Corsica.
    """)
    return


@app.cell
def _(plot_polygon, quickpaver):
    corsica = quickpaver.load_corsica_contour()
    france = quickpaver.load_france_contour()
    france_and_corsica = quickpaver.load_france_and_corsica_contour()
    plot_polygon(france_and_corsica, add_points=False)
    return corsica, france, france_and_corsica


@app.cell
def _(mo):
    mo.md(r"""
    - Define a helper function to plot the results
    """)
    return


@app.cell
def _(Optional, Union, ngp, plot_polygon, shapely):
    def plot_helper(
        grid: shapely.MultiPolygon,
        surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
        plotter: Optional[ngp.Plotter] = None,
    ):
        if plotter is None:
            plotter = ngp.Plotter()
        ax = plotter.axes[0]
        plot_polygon(surface_to_cover, ax=ax, add_points=False, color="r")
        plot_polygon(
            grid,
            ax=ax,
            add_points=False,
        )
        ax.set_aspect("equal")
        plotter.close()
        return plotter.fig

    return (plot_helper,)


@app.cell
def _(mo):
    mo.md(r"""
    - Start with a square tiling, without rotation and no anisotropy.
    We can see that only the meshes intersecting the surface to be covered are retained.
    """)
    return


@app.cell
def _(corsica, plot_helper, quickpaver):
    grid_squares_corsica_no_rot_no_ani, _adj = quickpaver.gen_polygonal_tiling(
        corsica,
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
    )
    plot_helper(grid_squares_corsica_no_rot_no_ani, corsica)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - To obtain extended coverage, you simply need to modify the working domain.
    For example, you can choose the minimum bounding ball:
    """)
    return


@app.cell
def _(corsica, plot_helper, quickpaver, shapely):
    grid_squares_corsica_circle, _adj = quickpaver.gen_polygonal_tiling(
        shapely.minimum_bounding_circle(corsica).buffer(50.0),
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
    )
    plot_helper(grid_squares_corsica_circle, corsica)
    return (grid_squares_corsica_circle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or using the bounding box
    """)
    return


@app.cell
def _(corsica, plot_helper, quickpaver, shapely):
    grid_squares_corsica_rectangle, _adj = quickpaver.gen_polygonal_tiling(
        shapely.box(*corsica.bounds),
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
    )
    plot_helper(grid_squares_corsica_rectangle, corsica)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that by default, the paving is always aligned in (0, 0), i.e.,
    for an infinite paving, one tile center is (0, 0)
    """)
    return


@app.cell
def _(corsica, grid_squares_corsica_circle, ngp, plot_helper):
    pl = ngp.Plotter()
    plot_helper(grid_squares_corsica_circle, corsica, plotter=pl)
    pl.axes[0].scatter(0.0, 0.0, label="Alignment point (0, 0)", c="red")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It can be modified to "align" the grid with a given center
    """)
    return


@app.cell
def _(corsica, ngp, plot_helper, quickpaver, shapely):
    pl2 = ngp.Plotter()
    grid_squares_corsica_circle2, _adj2 = quickpaver.gen_polygonal_tiling(
        shapely.minimum_bounding_circle(corsica).buffer(50.0),
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
        alignment_point=(50.0, 50.0),
    )
    plot_helper(grid_squares_corsica_circle2, corsica, plotter=pl2)
    pl2.axes[0].scatter(0.0, 0.0, label="Default alignment point (0, 0)", c="red")
    pl2.axes[0].scatter(50.0, 50.0, label="New alignment point", c="green")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It works even if the point lies outside the surface
    """)
    return


@app.cell
def _(corsica, ngp, plot_helper, quickpaver, shapely):
    pl3 = ngp.Plotter()
    grid_squares_corsica_circle3, _adj3 = quickpaver.gen_polygonal_tiling(
        shapely.minimum_bounding_circle(corsica).buffer(50.0),
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
        alignment_point=(-500.0, 1025.0),
    )
    plot_helper(grid_squares_corsica_circle3, corsica, plotter=pl3)
    pl3.axes[0].scatter(0.0, 0.0, label="Default alignment point (0, 0)", c="red")
    pl3.axes[0].scatter(-500, 1025.0, label="New alignment point", c="green")
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Now, let's try with anisotropy and rotation (it goes clockwise)
    """)
    return


@app.cell
def _(corsica, plot_helper, quickpaver):
    grid_squares_corsica_rot_ani, grid_squares_corsica_rot_ani_adj = (
        quickpaver.gen_polygonal_tiling(
            corsica,
            poly_type=quickpaver.PolygonType.RECTANGLE,
            edge_length=100.0,
            anisotropy_ratio=2.0,
            rot_deg=30.0,
        )
    )
    plot_helper(grid_squares_corsica_rot_ani, corsica)
    return grid_squares_corsica_rot_ani, grid_squares_corsica_rot_ani_adj


@app.cell
def _(mo):
    mo.md(r"""
    - Let's try again with different parameters
    """)
    return


@app.cell
def _(corsica, plot_helper, quickpaver):
    grid_squares_corsica_rot_ani2, _adj = quickpaver.gen_polygonal_tiling(
        corsica,
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=0.5,
        rot_deg=-30.0,
    )
    plot_helper(grid_squares_corsica_rot_ani2, corsica)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - As previously introduced, hexagonal grids are also supported. Let's now play
    with France outline:
    """)
    return


@app.cell
def _(france, plot_helper, quickpaver):
    grid_hexagons_france_no_rot_no_ani, _adj = quickpaver.gen_polygonal_tiling(
        france,
        poly_type=quickpaver.PolygonType.HEXAGON,
        edge_length=100.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
    )
    plot_helper(grid_hexagons_france_no_rot_no_ani, france)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - By default the hexagons are "flat-top" oriented, but it is very easily changed:
    """)
    return


@app.cell
def _(france, plot_helper, quickpaver):
    grid_hexagons_france_rot, adj_grid_hexagons_france_rot = (
        quickpaver.gen_polygonal_tiling(
            france,
            poly_type=quickpaver.PolygonType.HEXAGON,
            edge_length=500.0,
            anisotropy_ratio=1.0,
            rot_deg=30.0,
        )
    )
    plot_helper(grid_hexagons_france_rot, france)
    return adj_grid_hexagons_france_rot, grid_hexagons_france_rot


@app.cell
def _(mo):
    mo.md(r"""
    Now let's try with triangles
    """)
    return


@app.cell
def _(france_and_corsica, plot_helper, quickpaver):
    grid_triangles_rot_no_ani, _adj = quickpaver.gen_polygonal_tiling(
        france_and_corsica,
        poly_type=quickpaver.PolygonType.TRIANGLE,
        edge_length=550.0,
        anisotropy_ratio=1.0,
        rot_deg=-30.0,
    )
    plot_helper(grid_triangles_rot_no_ani, france_and_corsica)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Same with anisotropy:
    """)
    return


@app.cell
def _(france_and_corsica, plot_helper, quickpaver):
    grid_triangles_no_rot_ani, _adj = quickpaver.gen_polygonal_tiling(
        france_and_corsica,
        poly_type=quickpaver.PolygonType.TRIANGLE,
        edge_length=500.0,
        anisotropy_ratio=3.0,
        rot_deg=0.0,
    )
    plot_helper(grid_triangles_no_rot_ani, france_and_corsica)
    return


@app.cell
def _(france_and_corsica, plot_helper, quickpaver):
    grid_triangles_rot_ani, adj_grid_triangles_rot_ani = (
        quickpaver.gen_polygonal_tiling(
            france_and_corsica,
            poly_type=quickpaver.PolygonType.TRIANGLE,
            edge_length=1000.0,
            anisotropy_ratio=1.5,
            rot_deg=45.0,
        )
    )
    plot_helper(grid_triangles_rot_ani, france_and_corsica)
    return adj_grid_triangles_rot_ani, grid_triangles_rot_ani


@app.cell
def _(mo):
    mo.md(r"""
    - It is also possible to extract both centers and vertices (the adjacency between
    the vertices and the centers is also provided)
    """)
    return


@app.cell
def _(france, grid_hexagons_france_rot, plot_helper, quickpaver):
    centers = quickpaver.extract_tiling_centers(grid_hexagons_france_rot.geoms)
    vertices, v_c_adj, clusters, _ = quickpaver.extract_tiling_vertices(
        grid_hexagons_france_rot.geoms
    )

    plotter2 = plot_helper(grid_hexagons_france_rot, france)
    plotter2.axes[0].scatter(centers[:, 0], centers[:, 1], color="b", label="centers")
    plotter2.axes[0].scatter(
        vertices[:, 0], vertices[:, 1], color="g", label="vertices"
    )
    plotter2.axes[0].legend()
    plotter2
    return (centers,)


@app.cell
def _(
    adj_grid_hexagons_france_rot,
    centers,
    france,
    grid_hexagons_france_rot,
    plot_helper,
    quickpaver,
):
    plotter3 = plot_helper(grid_hexagons_france_rot, france)
    plotter3.axes[0].scatter(centers[:, 0], centers[:, 1], color="b", label="centers")
    quickpaver.draw_adjacency(
        plotter3.axes[0],
        centers,
        adj_grid_hexagons_france_rot,
        color="red",
        label="adjacency",
    )

    plotter3.axes[0].legend()
    plotter3
    return


@app.cell
def _(france_and_corsica, grid_triangles_rot_ani, plot_helper, quickpaver):
    centers2 = quickpaver.extract_tiling_centers(grid_triangles_rot_ani.geoms)
    vertices2, v_c_adj2, clusters_2, _ = quickpaver.extract_tiling_vertices(
        grid_triangles_rot_ani.geoms
    )

    plotter4 = plot_helper(grid_triangles_rot_ani, france_and_corsica)
    plotter4.axes[0].scatter(centers2[:, 0], centers2[:, 1], color="b", label="centers")
    plotter4.axes[0].scatter(
        vertices2[:, 0], vertices2[:, 1], color="g", label="vertices"
    )
    plotter4.axes[0].legend()
    plotter4
    return (centers2,)


@app.cell
def _(
    adj_grid_triangles_rot_ani,
    centers2,
    france_and_corsica,
    grid_triangles_rot_ani,
    plot_helper,
    quickpaver,
):
    plotter5 = plot_helper(grid_triangles_rot_ani, france_and_corsica)
    plotter5.axes[0].scatter(centers2[:, 0], centers2[:, 1], color="b", label="centers")
    quickpaver.draw_adjacency(
        plotter5.axes[0],
        centers2,
        adj_grid_triangles_rot_ani,
        color="red",
        label="adjacency",
    )
    plotter5.axes[0].legend()
    plotter5
    return


@app.cell
def _(corsica, grid_squares_corsica_rot_ani, plot_helper, quickpaver):
    centers3 = quickpaver.extract_tiling_centers(grid_squares_corsica_rot_ani.geoms)
    vertices3, v_c_adj3, clusters_3, _ = quickpaver.extract_tiling_vertices(
        grid_squares_corsica_rot_ani.geoms
    )

    plotter6 = plot_helper(grid_squares_corsica_rot_ani, corsica)
    plotter6.axes[0].scatter(centers3[:, 0], centers3[:, 1], color="b", label="centers")
    plotter6.axes[0].scatter(
        vertices3[:, 0], vertices3[:, 1], color="g", label="vertices"
    )
    plotter6.axes[0].legend()
    plotter6
    return centers3, plotter6


@app.cell
def _(centers3, grid_squares_corsica_rot_ani_adj, plotter6, quickpaver):
    quickpaver.draw_adjacency(
        plotter6.axes[0],
        centers3,
        grid_squares_corsica_rot_ani_adj,
        color="red",
        label="adjacency",
    )
    plotter6.axes[0].legend()
    plotter6
    return


@app.cell
def _(mo):
    mo.md(r"""
    Holes are supported as well.
    """)
    return


@app.cell
def _(plot_helper, quickpaver, shapely):
    donut = shapely.Point((0.0, 0.0)).buffer(200.0) - shapely.Point((0.0, 0.0)).buffer(
        100.0
    )

    grid_hex_donut, _adj = quickpaver.gen_polygonal_tiling(
        donut,
        poly_type=quickpaver.PolygonType.HEXAGON,
        edge_length=10.001,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
    )
    plot_helper(grid_hex_donut, donut)
    return


if __name__ == "__main__":
    app.run()
