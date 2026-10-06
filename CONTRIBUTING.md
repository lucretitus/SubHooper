# Contributing

Focused bug reports and pull requests are welcome. Reports can contain filenames and subtitle text; remove personal content before sharing diagnostics.

## Source setup

Windows x64 is the supported build platform. From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\app\start-development.ps1
```

The launcher prepares the development tools when needed and selects an available loopback port. Published installers use bundled application assets and do not need this development server.

## Checks

From `app/gui`:

```powershell
npm ci
npm test
npm run build
cargo test --locked --manifest-path src-tauri/Cargo.toml
```

From the repository root, with Python 3.11 or newer:

```powershell
python -m pip install numpy==2.2.6
python -m unittest discover -s app/tests -v
```

## Releases

The `v0.4.3` tag must match the version in the application configuration. The release workflow runs Windows checks and creates a draft installer release with signed update artifacts. Publish the draft after testing the installer.

Keep the existing updater public key. Signing secrets belong in GitHub Actions secrets, never in the repository. Do not commit models, downloaded runtimes, build output, videos, reports, results, or private keys.
