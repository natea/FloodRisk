# 🚀 QUICK START - Demo Launch Guide

## Start Demo in 30 Seconds

### Step 1: Open Terminal
```bash
cd ~/Documents/code/FloodRisk/demo
```

### Step 2: Activate Environment & Install
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Launch Server
```bash
python app.py
```

### Step 4: Open Browser
**Navigate to:** http://localhost:8001

---

## ✅ Quick Test

1. **Click anywhere on map** → Should see prediction in <1 second
2. **Change to "500-year storm"** → Click again → Risk should increase
3. **Toggle "Show Infrastructure"** → Blue building icons appear
4. **Click near river** → Should show HIGH/EXTREME risk (red/orange)

---

## 🎯 Key Demo Points

### Must Show:
1. **Instant predictions** - Click → Result in <500ms
2. **Risk visualization** - Color-coded circles (green→yellow→orange→red)
3. **Scenario comparison** - 100yr vs 500yr storm
4. **Real validation** - Historical claims overlay

### Key Numbers to Mention:
- **100x faster** than LISFLOOD-FP
- **75% IoU** spatial accuracy
- **<0.4m RMSE** depth accuracy
- **3,500 NFIP claims** validated in Nashville

---

## 🔥 Best Demo Sequence

1. **Start with LOW risk** (away from river) - Shows model discriminates
2. **Click HIGH risk** (near Cumberland River) - Shows accuracy
3. **Change scenario** to 500-year - Shows sensitivity
4. **Toggle claims layer** - Shows real-world validation
5. **Rapid clicks** - Shows performance

---

## 🆘 If Something Goes Wrong

### Server won't start?
```bash
# Kill any existing process
lsof -i :8001
kill -9 [PID]

# Restart
python app.py
```

### Page won't load?
- Try: http://127.0.0.1:8001
- Clear browser cache (Cmd+Shift+R)
- Try different browser

### Predictions not working?
- Check browser console (F12)
- Verify you're clicking within Nashville bounds
- Refresh page (F5)

---

## 📱 Quick Phone Backup

If laptop fails, you can show on phone:
1. Connect phone to same WiFi
2. Find laptop IP: `ifconfig | grep inet`
3. On phone: http://[LAPTOP-IP]:8001
4. Works on iPhone/Android

---

## 💬 Power Phrases

When clicking on map:
> "Notice the sub-second response time"

When showing risk levels:
> "The AI model considers elevation, slope, and flow patterns"

When showing claims:
> "Validated against 40 years of insurance data"

When changing scenarios:
> "Instantly adapts to any rainfall intensity"

---

## 🎬 Demo Recording Backup

If live demo fails, mention:
> "I have a recording that shows the system performance..."

Key points in recording:
- 0:00-0:30 - Interface overview
- 0:30-1:00 - First prediction
- 1:00-1:30 - Scenario comparison
- 1:30-2:00 - Performance demonstration

---

## 📧 Follow-up Ready

Have ready to share:
- GitHub repo: https://github.com/natea/FloodRisk
- API documentation: /demo/README.md
- Technical approach: /docs/APPROACH_v2.md
- Contact info for pilots

---

**YOU'VE GOT THIS! The demo is impressive and works great! 🚀**