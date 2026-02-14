# PocketPy WASM NPM Packaging Strategy

## Background

`make build_wasm` compiles pocketpy (with numpy bindings) to WebAssembly, producing:

- `pocketpy.js` (~64KB) -- Emscripten glue code, supports both Node.js and browser
- `pocketpy.wasm` (~2.4MB) -- WebAssembly binary

We want to publish this as npm packages, with 3 tiers of runtime:

1. `@pocketpy/core` -- raw pocketpy (lz4, msgpack, etc.)
2. `@pocketpy/numpy` -- pocketpy + numpy (this repo)
3. `@pocketpy/geos` -- pocketpy + geometry libs (for pro users)

Users should be able to switch between runtimes dynamically at runtime.

## Why Not Webpack / Treeshake?

WASM packages should **not** go through webpack/rollup treeshaking. Reasons:

- `.wasm` is an opaque binary -- bundlers can't treeshake it, they can only copy it as an asset
- 2.4MB of WASM binary mixed into a JS bundle is terrible for performance
- Emscripten glue code has side effects and global state (`Module`, `FS`, heap views...), bundlers handle it poorly
- Users of this library are running a **Python interpreter** -- they need the whole runtime, not a subset

The primary usage pattern is: **load the WASM separately in the browser** (via `<script>` tag or dynamic fetch from CDN).

This pattern is battle-tested by `sql.js` (~3M weekly npm downloads), `pyodide`, `ffmpeg.wasm`, etc. The key insight: **npm is just a distribution mechanism, not a build mechanism** for WASM packages.

## Package Architecture

```
@pocketpy/core    <-- pure assets (pocketpy.js + pocketpy.wasm)
@pocketpy/numpy   <-- pure assets (pocketpy.js + pocketpy.wasm)
@pocketpy/geos    <-- pure assets (pocketpy.js + pocketpy.wasm)

@pocketpy/registry <-- tiny JS-only package (~2KB)
                      no dependency on the above three
                      loads them from CDN at runtime
```

The registry package has **zero npm dependencies** on the variant packages. It resolves variant names to CDN URLs at runtime. Users don't `npm install` all three -- they install only the registry, and the right WASM binary is fetched on demand.

## Variant Package Structure

Each variant package (core / numpy / geos) has the same layout:

```
@pocketpy/numpy/
  package.json
  dist/
    pocketpy.js        # Emscripten glue (as-is from build)
    pocketpy.wasm      # WASM binary (as-is from build)
```

### `package.json` (variant)

```json
{
  "name": "@pocketpy/numpy",
  "version": "0.1.0",
  "description": "PocketPy with NumPy support (WebAssembly)",
  "exports": {
    "./pocketpy.js": "./dist/pocketpy.js",
    "./pocketpy.wasm": "./dist/pocketpy.wasm"
  },
  "files": ["dist/"],
  "keywords": ["python", "wasm", "numpy", "pocketpy"]
}
```

These packages are **pure asset packages** -- no JS entry point, no `main` field. They exist solely so that:

1. CDN services (unpkg, jsdelivr) can serve the files
2. Users can `npm install` them if they want local copies
3. Versions are tracked in the npm registry

## Registry Package

`@pocketpy/registry` is a tiny JS package that provides:

- A unified `createPocketPy(variant)` API
- CDN resolution for variant packages
- Custom variant registration
- CDN override

### `package.json` (registry)

```json
{
  "name": "@pocketpy/registry",
  "version": "0.1.0",
  "description": "Runtime loader for PocketPy WASM variants",
  "type": "module",
  "main": "./dist/registry.mjs",
  "exports": {
    ".": {
      "import": "./dist/registry.mjs",
      "require": "./dist/registry.cjs"
    }
  },
  "files": ["dist/"],
  "keywords": ["python", "wasm", "pocketpy"]
}
```

### Registry Implementation

```js
// @pocketpy/registry

const REGISTRY = {
  core:  { pkg: '@pocketpy/core',  version: '0.1.0' },
  numpy: { pkg: '@pocketpy/numpy', version: '0.1.0' },
  geos:  { pkg: '@pocketpy/geos',  version: '0.1.0' },
};

let cdnTemplate = (pkg, version, file) =>
  `https://unpkg.com/${pkg}@${version}/dist/${file}`;

export function setCDN(templateFn) {
  cdnTemplate = templateFn;
}

export function register(name, { pkg, version }) {
  REGISTRY[name] = { pkg, version };
}

export async function createPocketPy(name = 'core', options = {}) {
  const entry = REGISTRY[name];
  if (!entry) throw new Error(`Unknown pocketpy variant: "${name}"`);

  const { pkg, version } = entry;
  const baseURL = options.baseURL || cdnTemplate(pkg, version, '');

  const Module = await loadModule(baseURL, options);

  Module.ccall('py_initialize', null, [], []);
  Module.ccall('py_module_initialize', 'boolean', [], []);

  return {
    variant: name,
    exec(code, filename = 'main.py') {
      const ok = Module.ccall(
        'py_exec', 'boolean',
        ['string', 'string', 'number', 'number'],
        [code, filename, 0, 0]
      );
      if (!ok) {
        Module.ccall('py_printexc', null, [], []);
        Module.ccall('py_clearexc', null, ['number'], [0]);
      }
      return ok;
    },
    destroy() {
      Module.ccall('py_finalize', null, [], []);
    },
    _module: Module,
  };
}

function loadModule(baseURL, options) {
  return new Promise((resolve, reject) => {
    if (typeof window !== 'undefined') {
      loadModuleBrowser(baseURL, options, resolve, reject);
    } else {
      loadModuleNode(baseURL, options, resolve, reject);
    }
  });
}

function loadModuleBrowser(baseURL, options, resolve, reject) {
  const Module = {
    locateFile: (path) => baseURL + path,
    print: options.print || console.log,
    printErr: options.printErr || console.error,
    onRuntimeInitialized() { resolve(Module); },
  };

  const script = document.createElement('script');
  script.src = baseURL + 'pocketpy.js';
  script.onerror = () => reject(new Error(`Failed to load ${script.src}`));

  const prev = window.Module;
  window.Module = Module;
  script.onload = () => { window.Module = prev; };

  document.head.appendChild(script);
}

async function loadModuleNode(baseURL, options, resolve, reject) {
  try {
    const fs = await import('node:fs');

    let jsCode;
    if (baseURL.startsWith('http')) {
      const resp = await fetch(baseURL + 'pocketpy.js');
      jsCode = await resp.text();
    } else {
      jsCode = fs.readFileSync(baseURL + 'pocketpy.js', 'utf-8');
    }

    const Module = {
      locateFile: (p) => baseURL + p,
      print: options.print || console.log,
      printErr: options.printErr || console.error,
      onRuntimeInitialized() { resolve(Module); },
    };

    const fn = new Function('Module', jsCode);
    fn(Module);
  } catch (e) {
    reject(e);
  }
}
```

## Usage Examples

### Browser: Script Tag (simplest)

```html
<script src="https://unpkg.com/@pocketpy/registry/dist/registry.umd.js"></script>
<script>
  PocketPy.createPocketPy('numpy').then(py => {
    py.exec('import numpy as np; print(np.eye(3))');
  });
</script>
```

### Browser: ESM

```js
import { createPocketPy } from '@pocketpy/registry';

const variant = new URLSearchParams(location.search).get('rt') || 'numpy';
const py = await createPocketPy(variant);
py.exec('import numpy as np; print(np.zeros(5))');
py.destroy();
```

### Self-Hosted CDN

```js
import { createPocketPy, setCDN } from '@pocketpy/registry';

setCDN((pkg, version, file) =>
  `https://my-cdn.example.com/pocketpy/${pkg}/${version}/${file}`
);

const py = await createPocketPy('geos');
```

### Custom Variant Registration

```js
import { createPocketPy, register } from '@pocketpy/registry';

register('numpy-nightly', {
  pkg: '@pocketpy/numpy',
  version: '0.2.0-nightly.3',
});

const py = await createPocketPy('numpy-nightly');
```

### Direct Variant Usage (without registry)

Users can also skip the registry and use a variant package directly via CDN:

```html
<script>
  var Module = {
    locateFile: (path) => `https://unpkg.com/@pocketpy/numpy@0.1.0/dist/${path}`,
    onRuntimeInitialized() {
      Module.ccall('py_initialize', null, [], []);
      Module.ccall('py_module_initialize', 'boolean', [], []);
      Module.ccall('py_exec', 'boolean',
        ['string', 'string', 'number', 'number'],
        ['print("hello")', 'main.py', 0, 0]);
    },
    print: console.log,
    printErr: console.error,
  };
</script>
<script src="https://unpkg.com/@pocketpy/numpy@0.1.0/dist/pocketpy.js"></script>
```

## Summary

| Concern | Answer |
|---|---|
| Webpack / bundlers? | Users **don't** bundle the `.wasm`. They serve it as a static asset or load from CDN. |
| Treeshaking? | Not applicable. The whole runtime is needed. |
| CDN / script tag? | Works out of the box via `unpkg.com/@pocketpy/*/dist/pocketpy.js` |
| npm install? | Works via registry package or direct variant import |
| Swappable runtimes? | `createPocketPy('numpy')` / `createPocketPy('geos')` -- one line change, runtime dynamic |
| Registry dependencies? | Zero. Registry is JS-only, fetches WASM from CDN on demand. |
