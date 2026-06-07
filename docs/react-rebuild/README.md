# Rebuilding SDF Modeler in React Native + WebGPU

> A phased, beginner-friendly guide to understanding the existing Rust app **and** re-creating it as a mobile app with **React Native + Expo**, **`react-native-webgpu`**, and **TypeGPU**.

## 📖 The whole guide lives in one file

➡️ **[sdf-modeler-react-rebuild.md](sdf-modeler-react-rebuild.md)**

It's written as a top-to-bottom course (with a table of contents you can jump around in), aimed at someone who knows **React/TypeScript** but is **new to Rust and WebGPU**. Every concept is built up with plain language, TypeScript, and Mermaid diagrams.

## What's inside

| # | Section | After it you can… |
| --- | --- | --- |
| 00 | Orientation — the mental model | Explain what an SDF modeler is to anyone |
| 01 | WebGPU + Expo setup from zero | Run a WebGPU app on your phone that fills the screen with color |
| 02 | Your first shader | Write a program that runs on every pixel |
| 03 | Raymarching 101 | Render a lit 3D sphere from a math formula |
| 04 | The scene data model | Represent a multi-shape scene as TypeScript data |
| 05 | Runtime codegen ⭐ | Generate a shader from your scene data at runtime |
| 06 | Buffers, camera & the render loop | Orbit, pan, and zoom with touch |
| 07 | Picking & selection | Tap a shape to select it |
| 08 | Sculpting & voxels | Sculpt geometry like clay |
| 09 | React architecture & practices | Structure the whole app the right way |
| — | Glossary | Look up any unfamiliar term |

⭐ Section 05 (runtime codegen) is the heart of the app.

## Scope

Goes **deep** on the rendering core and sculpting; deliberately **defers** mesh export, advanced lighting, and the full desktop UI (all named in section 09, not forgotten). This is a **re-imagining, not a 1:1 port** — it simplifies where a beginner benefits and says so.

## ⚠️ Before you start

- Targets **mobile** (iOS + Android) via **Expo**.
- `react-native-webgpu` needs **React Native 0.81+** and the **New Architecture**, and **does not run in Expo Go** — you'll build a **custom dev client** (covered in section 01).
- **SDF raymarching is heavy on a phone.** Performance is a recurring theme, with cost-control levers taught throughout.

**Start reading: [sdf-modeler-react-rebuild.md »](sdf-modeler-react-rebuild.md)**
