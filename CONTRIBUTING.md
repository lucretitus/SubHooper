# Contributing

Bug reports and focused pull requests are welcome.

## Development checks

From `app/gui`:

```powershell
npm ci
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml
```

From the repository root, with Python 3.14 available:

```powershell
python -m unittest discover -s app/tests -v
```

Do not commit downloaded AI models, VideoSubFinder, Python runtimes, OCR package
environments, API keys, updater private keys, videos, extracted frames, results,
or diagnostic reports.

