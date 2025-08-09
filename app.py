import os, asyncio, signal
import ccxt, pandas as pd
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message
from aiogram.filters import Command
from fastapi import FastAPI
import uvicorn

# ---------- ENV ----------
API_KEY   = os.getenv("BYBIT_API_KEY")
API_SECRET= os.getenv("BYBIT_API_SECRET")
TESTNET   = os.getenv("TESTNET","0") == "1"   # 0 = LIVE (боевой)
SYMBOL    = os.getenv("SYMBOL","BTCUSDT")     # USDT‑перпеты
LEVERAGE  = int(os.getenv("LEVERAGE","5"))
RISK      = float(os.getenv("RISK_PER_TRADE","0.05"))  # 5% на сделку
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID  = int(os.getenv("ADMIN_ID","0"))

TF = "5m"
EMAF, EMAS = 12, 48

# ---------- EXCHANGE ----------
def mk_ex():
    ex = ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}  # USDT‑перпеты
    })
    if TESTNET:
        ex.set_sandbox_mode(True)
    return ex

ex = mk_ex()
ex.load_markets()

# плечо (one‑way режим)
try:
    ex.set_leverage(LEVERAGE, SYMBOL, params={"market":"linear"})
except Exception:
    pass

def fetch_ohlcv(limit=500):
    o = ex.fetch_ohlcv(SYMBOL, timeframe=TF, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
    return df

def ema(series, n): return pd.Series(series).ewm(span=n, adjust=False).mean()
def rsi(series, n=14):
    s = pd.Series(series); d = s.diff()
    u = d.clip(lower=0); v = -d.clip(upper=0)
    return 100 - 100/(1+(u.rolling(n).mean()/(v.rolling(n).mean().replace(0,1e-9))))

def gann_levels(p, step=0.005, n=4):
    ups=[p*(1+step*i) for i in range(1,n+1)]
    dns=[p*(1-step*i) for i in range(1,n+1)]
    return sorted(dns)+[p]+sorted(ups)

async def place(side, qty_usdt, px, sl, tp):
    q = round(qty_usdt/px, 6)
    side_in  = "buy"  if side=="long"  else "sell"
    side_out = "sell" if side=="long"  else "buy"
    base_params = {"reduce_only": False, "positionIdx": 0}  # one-way

    # вход
    o = ex.create_order(SYMBOL, "market", side_in, q, None, base_params)

    # защитные ордера (reduce_only)
    if sl:
        ex.create_order(
            SYMBOL, "market", side_out, None, None,
            {"stopPrice": float(sl), "reduce_only": True, "triggerDirection": 2 if side=="long" else 1, "positionIdx":0}
        )
    if tp:
        ex.create_order(
            SYMBOL, "market", side_out, None, None,
            {"stopPrice": float(tp), "reduce_only": True, "triggerDirection": 1 if side=="long" else 2, "positionIdx":0}
        )
    return o

def pos_size(balance_usdt, entry, sl, side):
    risk = max(5, balance_usdt * RISK)
    if not sl: sl = entry*(0.995 if side=="long" else 1.005)
    dist = max(abs(entry-sl), entry*0.0002)
    qty_usdt = risk / dist
    return round(qty_usdt, 2), sl

# ---------- TRADING LOOP ----------
_running = False
async def trader(bot: Bot):
    global _running
    _running = True
    await bot.send_message(ADMIN_ID, f"Bybit LIVE: {SYMBOL}, риск {int(RISK*100)}% — старт.")

    in_pos = False
    while _running:
        try:
            bal = ex.fetch_balance().get("USDT",{}).get("free", 0.0)
            df = fetch_ohlcv()
            c = df["close"]
            emaf, emas = ema(c,EMAF), ema(c,EMAS)
            r = rsi(c,14)
            px = float(c.iloc[-1])

            bull = emaf.iloc[-2]<=emas.iloc[-2] and emaf.iloc[-1]>emas.iloc[-1] and r.iloc[-1]>50
            bear = emaf.iloc[-2]>=emas.iloc[-2] and emaf.iloc[-1]<emas.iloc[-1] and r.iloc[-1]<50
            near = any(abs(px-L)/px<0.0015 for L in gann_levels(px))

            if not in_pos:
                if bull and near:
                    qty, sl = pos_size(bal, px, px*0.995, "long")
                    tp = px*1.005
                    await place("long", qty, px, sl, tp)
                    in_pos=True
                    await bot.send_message(ADMIN_ID, f"LONG {SYMBOL} @ {px:.2f} | SL {sl:.2f} | TP {tp:.2f}")
                elif bear and near:
                    qty, sl = pos_size(bal, px, px*1.005, "short")
                    tp = px*0.995
                    await place("short", qty, px, sl, tp)
                    in_pos=True
                    await bot.send_message(ADMIN_ID, f"SHORT {SYMBOL} @ {px:.2f} | SL {sl:.2f} | TP {tp:.2f}")
            else:
                # если позиции нет (SL/TP закрыло) — снова ждём вход
                pos = ex.fetch_positions([SYMBOL])
                opened = any(abs(float(p.get("contracts",0) or 0))>0 for p in pos)
                if not opened:
                    in_pos=False
                    await bot.send_message(ADMIN_ID, "Позиция закрыта (SL/TP или вручную).")
        except Exception as e:
            await bot.send_message(ADMIN_ID, f"Ошибка: {e}")
            await asyncio.sleep(5)
        await asyncio.sleep(10)

# ---------- TELEGRAM ----------
bot = Bot(BOT_TOKEN)
dp = Dispatcher()
rt = Router()
dp.include_router(rt)

@rt.message(Command("start"))
async def start(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    await m.answer("Команды:\n/start_bot\n/stop_bot\n/status\n/symbol BTCUSDT\n/risk 5")

@rt.message(Command("start_bot"))
async def start_bot(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    global _running
    if _running: return await m.answer("Уже работает.")
    asyncio.create_task(trader(bot))
    await m.answer("Запускаю…")

@rt.message(Command("stop_bot"))
async def stop_bot(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    global _running
    _running=False
    await m.answer("Остановлен (жду завершения цикла).")

@rt.message(Command("status"))
async def status(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    bal = ex.fetch_balance().get("USDT",{}).get("free", 0.0)
    await m.answer(f"Пара: {SYMBOL}\nРиск: {int(RISK*100)}%\nБаланс: {bal:.2f} USDT\nРаботает: {bool(_running)}")

@rt.message(F.text.startswith("/symbol"))
async def set_symbol(m: Message):
    global SYMBOL
    if m.from_user.id!=ADMIN_ID: return
    parts = m.text.split()
    if len(parts)>=2:
        SYMBOL = parts[1].upper()
        await m.answer(f"OK. SYMBOL = {SYMBOL}")

@rt.message(F.text.startswith("/risk"))
async def set_risk(m: Message):
    global RISK
    if m.from_user.id!=ADMIN_ID: return
    parts = m.text.split()
    if len(parts)>=2:
        RISK = float(parts[1])/100
        await m.answer(f"OK. Риск = {int(RISK*100)}%")

# ---------- FastAPI (health) ----------
api = FastAPI()
@api.get("/health")
def health(): return {"ok":True}

async def main():
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, lambda: None)
    asyncio.create_task(dp.start_polling(bot))  # long polling
    config = uvicorn.Config(api, host="0.0.0.0", port=int(os.getenv("PORT","8080")), log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
