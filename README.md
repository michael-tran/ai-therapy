# AI Chat
Purpose is to allow an AI model to run on the client side using only javascript/typescript. The main technology used is onnx runtime. 

## Limitations
Since it is using the `wasm` which is bounded to the CPU the largest model one can use is about 2GB total, including any data required to be passed to the model. This limits which models can be used. `webgpu` and `webgl` can work to increase model size and faster runtime but I was unsuccessful in making this work and those two interfaces would require hardware specific to make it work.

`wasm` allow for multithreading but I would need to enable `crossOriginIsolated` to be true. Using vite locally and enabling it is not bad. However for github I needed to use a workaround which is `coi-serviceworker` under the `index.html` file.

Another limitation is github and the 50MB file size limit. Without paying money for github LFS I am limited to each file being less than 50MB. A single model even the smallest is several MB. So the python script here takes a single file and splits them up into smaller chunks for later processing. Which is why they are in parts.

## Future improvements

* I want to get the `webgpu` working eventually
* Enable message history but this also slow down the inference session a lot so maybe making a smarter way to cache or compress it would be better. Possible ideas would be KV caching in the `ai.worker.ts` instead of passing the whole message queue again.
* Getting an actually therapy chat bot working. Original I wanted a AI therapy bot to enable privacy and client side protection from big corp. but that dreams seems to be a ways since there is no small enough AI model that would work.
* Allow the downloading of message history and restore chat function for future use.


# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
