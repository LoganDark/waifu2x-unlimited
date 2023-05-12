# unlimited:waifu2x

Actually, a modified version that we made based on a reverse-engineered version
of the original at <https://unlimited.waifu2x.net>.

## Development

Use a TypeScript IDE like [WebStorm](https://www.jetbrains.com/webstorm).

### Code style

No double-quoted strings, use tabs for indentation, don't use jquery.

## Compiling

### WebStorm

Right-click `tsconfig.json` and choose `Compile TypeScript`.

You can configure it to recompile automatically by opening Settings and enabling
`Languages & Frameworks` > `TypeScript` > `TypeScript language service` >
`Recompile on changes`.

### `tsc`

You can run `tsc` with no arguments to create `script.js` and `script.js.map`.

### [esbuild](https://esbuild.github.io/)

It might eventually be possible to use esbuild if you don't want to install
`npm` (for `tsc`) or WebStorm (expensive), but as of this writing, esbuild
[doesn't support](https://github.com/evanw/esbuild/issues/3100) this type of
project yet, and we're not willing to break compatibility with `tsc`.
<!--

Download esbuild from this link, replacing `win32-x64` with [the name of your
platform](https://esbuild.github.io/getting-started/#other-ways-to-install):
<https://registry.npmjs.org/@esbuild/win32-x64/-/win32-x64-0.17.18.tgz>

Then run esbuild with these flags: -->

## Running

This project won't work over the `file://` protocol, you have to use a local
HTTP server. This is because:

- Fetch requests can't be made to `file://` URLs, so ORT can't load its WASM
  (even if it could, it wouldn't be able to download the model files anyway)
- Web Workers can't be created from `file://` URLs, so the models can't run
- Some extra HTTP headers need to be sent for multithreading to work

### Multithreading headers

ORT supports parallel model execution, which can make waifu2x much faster. But
browsers only allow the use of `SharedArrayBuffer` if some extra HTTP headers
are sent by the server:

- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Opener-Policy: same-origin`

Without them, multithreading won't work.

### [miniserve](https://github.com/svenstaro/miniserve)

`miniserve . --index index.html --header 'Cross-Origin-Embedder-Policy: require-corp' --header 'Cross-Origin-Opener-Policy: same-origin'`

### esbuild

esbuild can run in "serve" mode, which exposes a local HTTP server that
automatically rebuilds the project on reload. Unfortunately, esbuild
[doesn't support](https://github.com/evanw/esbuild/issues/3100) this type of
project yet, and we're not willing to break compatibility with `tsc`.
