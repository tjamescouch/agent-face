# renderer

pluggable renderer system that loads and dispatches to render backends.

## state

- currently active renderer instance
- renderer name

## capabilities

- dynamically load renderer modules by name from a renderers directory
- initialize renderer with configuration (stream, port, options)
- dispatch frame objects to the active renderer
- cleanly close and release renderer resources

## interfaces

exposes:
- RendererManager class with use, render, close

depends on:
- renderer plugin modules (json, ansi, canvas)

## invariants

- render throws if no renderer is loaded
- close nullifies the renderer reference
- renderer plugins implement init/render/close contract
- each renderer module is a standalone ES module with a default export
