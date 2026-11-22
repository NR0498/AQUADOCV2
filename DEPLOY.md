# Deploying AquaDoc to Vercel

## What is already set up

| Item | State |
| --- | --- |
| Vercel project | `aquadoc`, linked from the repo root (`.vercel/project.json`) |
| Services | `vercel.json` runs the dashboard (`web/`, Next.js) and the API (`main.py`, FastAPI) as one deployment |
| Routing | `/api/*` goes to the API, everything else to the dashboard (same origin, no CORS) |
| Region | `fra1` (Frankfurt), close to the satellite imagery on Planetary Computer (West Europe) |
| Database | Neon Postgres, `DATABASE_URL` set for Production and Preview |
| Excluded from upload | `.vercelignore` drops data, research, training, tests and virtual environments |
| GitHub | Connected to `NR0498/AQUADOCV2`: pushing to `main` deploys production |
| Production URL | https://aquadoc-ivory.vercel.app |

## Option A: Deploy from your machine (recommended)

The CLI uploads your working folder as it is, so uncommitted changes are included. Run these from the repo root:

```powershell
cd C:\projects\AQUADOCV2

# 1. Confirm you're logged in to the right account (should print nayanraj083-2851)
vercel whoami

# 2. Run the backend tests and the dashboard build before uploading
.venv\Scripts\python -m pytest -q tests
cd web; npm run build; cd ..

# 3. Deploy a preview first (behind Vercel login, not public)
vercel deploy

# 4. Test the preview: it should return {"status":"ok",...,"pipeline_version":"water-pipeline-v4"}
vercel curl <preview-url>/api/health

# 5. Deploy to production (https://aquadoc-ivory.vercel.app)
vercel deploy --prod
```

If `vercel whoami` fails, run `vercel login` first. If the CLI is missing on a new machine, install it with `npm i -g vercel`.

## Option B: Deploy by pushing to GitHub

The Vercel project is connected to `NR0498/AQUADOCV2`, so a push to `main` deploys production automatically. This route only uses committed files, so commit everything first, including `aquadoc_core/known_outlines.json`, which the SRM lake's outline needs:

```powershell
cd C:\projects\AQUADOCV2
git add aquadoc_core/ tests/ web/components/ learning.md DEPLOY.md
git commit -m "Add Lakes near you, SRM lake outline, nearby lake search"
git push origin main
```

Pushes to any other branch create preview deployments.

## After deploying, check these on the live site

1. **Health:** https://aquadoc-ivory.vercel.app/api/health should show `water-pipeline-v4`.
2. **Featured lake:** analyse Hussain Sagar.
3. **Each location method:** run one analysis each from Search by name, Coordinates and Pin on map.
4. **Lakes near you:** location only works over https, which Vercel provides.
5. **Share link:** open one from a result in a private window.

## If something fails

- **Logs:** use the Vercel dashboard → `aquadoc` → **Logs**. On the Hobby plan, `vercel logs` from the CLI may return "outside the available retention".
- **Errors name the cause:** an import failure shows the missing module, and unreadable images show a short reason.
- **Roll back:** `vercel rollback`, or promote an earlier deployment from the dashboard.
- **Database:** check the connection with `vercel env ls`, where `DATABASE_URL` should be listed for Production and Preview.

## Good to know

- **Production runs the last deployed build:** until you deploy, https://aquadoc-ivory.vercel.app runs an old build where analyses fail.
- **First requests are slower:** the first analysis after a quiet period takes a few extra seconds while the server starts.
- **Rate limit:** 8 analyses per visitor per 10 minutes on each server instance. Change it in `RATE_LIMIT` in `aquadoc_core/api.py` if you need more.
- **Don't commit `.env.local`:** `vercel link` created it, and it holds the database credentials. It is already covered by `.gitignore`.

## Running locally

```powershell
cd C:\projects\AQUADOCV2\web
npm run dev:all
```

Then open http://localhost:3000 (use `localhost`, not `127.0.0.1`). This starts the API on port 8000 and the dashboard on port 3000; stopping it stops both.
