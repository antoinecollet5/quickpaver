import marimo

__generated_with = "0.20.4"
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
    from typing import Union

    import matplotlib.pyplot as plt
    import nested_grid_plotter as ngp
    import quickpaver
    import shapely
    from shapely.plotting import plot_polygon

    return Union, ngp, plot_polygon, plt, quickpaver, shapely


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
def _(Union, ngp, plot_polygon, shapely):
    def plot_helper(
        grid: shapely.MultiPolygon,
        surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
    ):
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
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Or using the bounding box
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


@app.cell
def _(mo):
    mo.md(r"""
    - Now, let's try with anisotropy and rotation (it goes clockwise)
    """)
    return


@app.cell
def _(corsica, plot_helper, quickpaver):
    grid_squares_corsica_rot_ani, _adj = quickpaver.gen_polygonal_tiling(
        corsica,
        poly_type=quickpaver.PolygonType.RECTANGLE,
        edge_length=100.0,
        anisotropy_ratio=2.0,
        rot_deg=30.0,
    )
    plot_helper(grid_squares_corsica_rot_ani, corsica)
    return


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
    grid_hexagons_france_rot, _adj = quickpaver.gen_polygonal_tiling(
        france,
        poly_type=quickpaver.PolygonType.HEXAGON,
        edge_length=500.0,
        anisotropy_ratio=1.0,
        rot_deg=30.0,
    )
    plot_helper(grid_hexagons_france_rot, france)
    return (grid_hexagons_france_rot,)


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
    grid_triangles_rot_ani, _adj = quickpaver.gen_polygonal_tiling(
        france_and_corsica,
        poly_type=quickpaver.PolygonType.TRIANGLE,
        edge_length=1000.0,
        anisotropy_ratio=1.5,
        rot_deg=45.0,
    )
    plot_helper(grid_triangles_rot_ani, france_and_corsica)
    return (grid_triangles_rot_ani,)


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
    vertices, v_c_adj, clusters_2 = quickpaver.extract_tiling_vertices(
        grid_hexagons_france_rot.geoms
    )

    plotter2 = plot_helper(grid_hexagons_france_rot, france)
    plotter2.axes[0].scatter(centers[:, 0], centers[:, 1], color="b", label="centers")
    plotter2.axes[0].scatter(
        vertices[:, 0], vertices[:, 1], color="g", label="vertices"
    )
    plotter2.axes[0].legend()
    plotter2
    return


@app.cell
def _(france_and_corsica, grid_triangles_rot_ani, plot_helper, quickpaver):
    centers2 = quickpaver.extract_tiling_centers(grid_triangles_rot_ani.geoms)
    vertices2, v_c_adj2, clusters_3 = quickpaver.extract_tiling_vertices(
        grid_triangles_rot_ani.geoms
    )

    plotter3 = plot_helper(grid_triangles_rot_ani, france_and_corsica)
    plotter3.axes[0].scatter(centers2[:, 0], centers2[:, 1], color="b", label="centers")
    plotter3.axes[0].scatter(
        vertices2[:, 0], vertices2[:, 1], color="g", label="vertices"
    )
    plotter3.axes[0].legend()
    plotter3
    return


@app.cell
def _(mo):
    mo.md(r"""
    Of course, it works with holes
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
        edge_length=10.0,
        anisotropy_ratio=1.0,
        rot_deg=0.0,
    )
    plot_helper(grid_hex_donut, donut)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Export to shapefile

    It is sometimes convenient to export to shapefiles.

    TODO.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""

    """)
    return


if __name__ == "__main__":
    app.run()
