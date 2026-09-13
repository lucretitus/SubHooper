# GitHub release procedure

## One-time repository setup

1. Create a **public** empty GitHub repository named `SubHooper`. Do not add a
   README, license, or `.gitignore` in GitHub because they are included here.
2. Run `Generate-Updater-Key.cmd` once on a trusted Windows computer. Keep
   `private\SubHooper.key` and its password in two secure backups. Losing either
   prevents installed copies from accepting future updates.
3. Run `Configure-GitHub.cmd` and enter the GitHub account name. The script puts
   the public key and the exact GitHub Releases endpoint in `tauri.conf.json`.
4. In the GitHub repository, open **Settings > Secrets and variables > Actions**
   and create:
   - `TAURI_SIGNING_PRIVATE_KEY`: the complete contents of
     `private\SubHooper.key`.
   - `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`: the password used when generating the
     key.
5. Upload the source with Git or GitHub Desktop. A command-line example:

```powershell
git init
git add .
git commit -m "Release SubHooper 0.3.6 beta"
git branch -M main
git remote add origin https://github.com/GITHUB_ACCOUNT/SubHooper.git
git push -u origin main
```

## Create beta 0.3.6

1. Ensure the `main` branch Actions check is green.
2. Create and push the exact release tag:

```powershell
git tag -a v0.3.6 -m "SubHooper 0.3.6 beta"
git push origin v0.3.6
```

3. Open **Actions > Windows release** and wait for completion.
4. Open the draft under **Releases** and confirm that the NSIS setup EXE,
   `latest.json`, and updater signature files are attached.
5. Download the setup EXE onto a clean Windows test account. Complete the test
   list in `docs/WINDOWS_BETA_ACCEPTANCE.md`, then publish the draft as the
   repository's **Latest** release. Keep “beta” in its title and notes, but do
   not select GitHub's **pre-release** checkbox: the configured
   `releases/latest/download/latest.json` update endpoint excludes pre-releases.

## Future updates

1. Change the same version in `app/VERSION.txt`, `app/gui/package.json`,
   `app/gui/src-tauri/Cargo.toml`, and `app/gui/src-tauri/tauri.conf.json`.
2. Update `CHANGELOG.md` and run the validation commands.
3. Commit, create the matching `vX.Y.Z` tag, and push it.
4. Publish the generated draft release. Existing installations can then use
   **Settings > Check for updates** to download the signed replacement.

Never regenerate the updater key for an ordinary release.
