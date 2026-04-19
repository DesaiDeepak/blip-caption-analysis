# DVC setup (S3 remote)

This project uses DVC (Data Version Control) for large files and model artifacts. Below are steps to initialize DVC locally, add data (we include an example `data/sample.csv`), and configure an S3 remote for sharing.

## Local quickstart

1. Install dvc (recommended in your project venv):

```bash
source .venv/bin/activate
pip install dvc[s3]
```

2. Initialize dvc in the repo (do this once locally):

```bash
dvc init
```

3. Add the example CSV (or your own data) to DVC:

```bash
dvc add data/sample.csv
# this creates data/sample.csv.dvc and adds the file to .gitignore
git add data/sample.csv.dvc .gitignore
git commit -m "dvc: track sample csv"
```

4. Configure S3 remote (replace bucket and path):

```bash
dvc remote add -d s3remote s3://your-bucket/path
# optionally set endpoint and region
# dvc remote modify s3remote endpointurl https://s3.amazonaws.com
```

5. Push data to remote (requires AWS credentials):

```bash
dvc push
```

## Where to put S3 credentials

For local use, `aws configure` or environment variables work. For CI (GitHub Actions), add credentials as repository secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- (optional) `AWS_DEFAULT_REGION`

Add them in GitHub: Repository → Settings → Secrets & variables → Actions → New repository secret.

## Example GitHub Actions snippet to pull DVC data

Add these steps to a workflow before running jobs that need data (this example installs dvc, configures remote, and runs `dvc pull`):

```yaml
- name: Install DVC (with S3 support)
  run: |
    python -m pip install --upgrade pip
    pip install dvc[s3]

- name: Configure AWS creds for DVC
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    AWS_DEFAULT_REGION: ${{ secrets.AWS_DEFAULT_REGION }}
  run: |
    # ensure AWS env vars are available to dvc
    echo "AWS credentials set for DVC"

- name: dvc pull
  run: |
    dvc pull -v
```

Notes:
- `dvc pull` will fetch tracked files from the configured remote (`dvc remote add` must have been run locally and committed; alternatively you can run `dvc remote add` in CI and `dvc remote modify` with any endpoint config).
- If your DVC remote uses a custom endpoint or non-AWS S3-compatible provider, add `dvc remote modify s3remote endpointurl <URL>` and commit the `.dvc/config` if desired.

## Switching remote or adding multiple remotes

You can have separate remotes for models and datasets. Use descriptive names (e.g., `s3models`, `s3data`) and set `-d` on the default remote.

## Troubleshooting

- Permission errors: ensure the AWS keys have `s3:PutObject` / `s3:GetObject` for the bucket path.
- If DVC complains about missing `.dvc` config in the repo, run `dvc init` locally and commit `.dvc/config`.

---

If you'd like, I can:
- Initialize a `dvc.yaml` pipeline stage (example: `prepare-data`) and commit it.
- Run `dvc remote add` locally and commit `.dvc/config` for your S3 remote (I can prepare the commands and a patch, but you must add secrets to GitHub and optionally run the final `dvc push`).
