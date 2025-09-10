# 🚀 Deploy to Vercel Instructions

## Prerequisites
1. Create a free Vercel account at https://vercel.com/signup
2. Install Vercel CLI (already done)

## Deployment Steps

### Option 1: Deploy via CLI (Recommended)

1. **Login to Vercel:**
```bash
npx vercel login
```
Enter your email and follow the authentication link.

2. **Deploy the project:**
```bash
npx vercel
```

When prompted:
- Set up and deploy: `Y`
- Which scope: Select your account
- Link to existing project: `N` (first time)
- Project name: `nashville-flood-risk` (or press enter for default)
- Directory: `.` (current directory)
- Override settings: `N`

3. **Production deployment:**
```bash
npx vercel --prod
```

### Option 2: Deploy via GitHub

1. Push your code to GitHub:
```bash
git add .
git commit -m "Add Vercel deployment configuration"
git push origin main
```

2. Go to https://vercel.com/new
3. Import your GitHub repository
4. Vercel will auto-detect the configuration
5. Click "Deploy"

## Your Demo URLs

After deployment, you'll get URLs like:
- Preview: `https://nashville-flood-risk-[hash].vercel.app`
- Production: `https://nashville-flood-risk.vercel.app`

## Environment Variables (Optional)

If you need environment variables, add them in Vercel dashboard:
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add any required variables

## Files Created for Vercel

- **vercel.json** - Deployment configuration
- **api/index.py** - Serverless function with embedded HTML
- **package.json** - Node.js configuration
- **.vercelignore** - Files to exclude from deployment

## Test Your Deployment

Once deployed, test the following endpoints:
- `/` - Main interactive map interface
- `/health` - API health check
- `/api/predict` - Flood risk predictions (POST)
- `/api/scenarios` - Available scenarios (GET)
- `/api/statistics` - Demo statistics (GET)

## Troubleshooting

### If deployment fails:
1. Check Python version (3.9+ required)
2. Verify all dependencies in requirements.txt
3. Check Vercel build logs for errors

### If API doesn't work:
1. Ensure CORS is properly configured
2. Check browser console for errors
3. Verify API routes in vercel.json

## Share Your Demo

Once deployed, share your Vercel URL with stakeholders:
- It's globally accessible
- Auto-scales with traffic
- Free tier supports your demo needs
- HTTPS enabled by default

## Local Testing Before Deploy

Test the Vercel function locally:
```bash
npx vercel dev
```
This runs the serverless function locally on port 3000.

---

**Ready to deploy!** Follow Option 1 or 2 above to get your demo live on Vercel.