import os, asyncio
import ccxt, pandas as pd
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message
from aiogram.filters import Command
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timezone

# ---------- ENV ----------
API_KEY    = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET    = os.getenv("TESTNET","0") == "1"       # 0 = LIVE
LEVERAGE   = int(os.getenv("LEVERAGE","5"))        # авто-плечо, будет выставляться ботом
RISK       = float(os.getenv("RISK_PER_TRADE","0.05"))  # доля депо на сделку
SYMBOLS    = [s.strip().upper() for s in os.getenv("SYMBOLS","BTCUSDT").split(",") if s.strip()]
HEDGE_MODE = os.getenv("HEDGE_MODE","1") == "1"

TRAIL_ENABLE       = os.getenv("TRAIL_ENABLE","1") == "1"
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT","0.4")) / 100.0
TRAIL_DISTANCE_PCT = float(os.getenv("TRAIL_DISTANCE_PCT","0.3")) / 100.0

BE_ENABLE          = os.getenv("BE_ENABLE","1") == "1"
BE_TRIGGER_PCT     = float(os.getenv("BE_TRIGGER_PCT","0.3")) / 100.0

EXIT_ON_OPPOSITE   = os.getenv("EXIT_ON_OPPOSITE","1") == "1"
REVERSE_ON_SIGNAL  = os.getenv("REVERSE_ON_SIGNAL","0") == "1"

DAILY_STOP_ENABLE  = os.getenv("DAILY_STOP_ENABLE","1") == "1"
DAILY_STOP_PCT     = float(os.getenv("DAILY_STOP_PCT","3")) / 100.0

BOT_TOKEN  = os.getenv("BOT_TOKEN")
ADMIN_ID   = int(os.getenv("ADMIN_ID","0"))
PROXY_URL  = os.getenv("PROXY_URL")

TF = "5m"
EMAF, EMAS = 12, 48

# ---------- EXCHANGE ----------
def mk_ex():
    ex = ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}  # USDT-перпеты (linear)
    })
    if TESTNET:
        ex.set_sandbox_mode(True)
    if PROXY_URL:
        ex.proxies = {'http': PROXY_URL, 'https': PROXY_URL}
        ex.aiohttp_proxy = PROXY_URL
    return ex

ex = mk_ex()
ex.load_markets()

# включим hedge-режим на стороне API (и руками в кабинете держи Hedge)
try:
    ex.set_position_mode(True if HEDGE_MODE else False)
except Exception:
    pass

# ---------- INDICATORS ----------
def fetch_ohlcv(symbol, limit=500):
    o = ex.fetch_ohlcv(symbol, timeframe=TF, limit=limit)
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

def pct(a, b):  # относительное изменение a к b
    return (a - b) / b

def pos_size(balance_usdt, entry, sl, side):
    risk = max(5, balance_usdt * RISK)  # минимум 5 USDT
    if not sl: sl = entry*(0.995 if side=="long" else 1.005)
    dist = max(abs(entry-sl), entry*0.0002)
    qty_usdt = risk / dist
    return round(qty_usdt, 2), sl

async def place(symbol, side, qty_usdt, px, sl, tp, positionIdx):
    """
    positionIdx: 1 = LONG, 2 = SHORT (hedge), 0 = one-way
    side: 'long'/'short'
    """
    # авто-плечо (x5) перед входом
    try:
        ex.set_leverage(LEVERAGE, symbol, params={"buyLeverage": LEVERAGE, "sellLeverage": LEVERAGE, "market":"linear"})
    except Exception:
        pass

    q = round(qty_usdt/px, 6)
    side_in  = "buy"  if side=="long"  else "sell"
    side_out = "sell" if side=="long"  else "buy"

    base = {"reduce_only": False, "positionIdx": positionIdx}
    o = ex.create_order(symbol, "market", side_in, q, None, base)

    trig_up   = 1 if side=="long" else 2
    trig_down = 2 if side=="long" else 1
    if sl:
        ex.create_order(symbol, "market", side_out, None, None,
                        {"stopPrice": float(sl), "reduce_only": True, "triggerDirection": trig_down, "positionIdx": positionIdx})
    if tp:
        ex.create_order(symbol, "market", side_out, None, None,
                        {"stopPrice": float(tp), "reduce_only": True, "triggerDirection": trig_up, "positionIdx": positionIdx})
    return o

async def place_protective(symbol, side, sl=None, tp=None, positionIdx=0):
    side_out = "sell" if side=="long" else "buy"
    trig_up   = 1 if side=="long" else 2
    trig_down = 2 if side=="long" else 1
    if sl:
        ex.create_order(symbol, "market", side_out, None, None,
                        {"stopPrice": float(sl), "reduce_only": True, "triggerDirection": trig_down, "positionIdx": positionIdx})
    if tp:
        ex.create_order(symbol, "market", side_out, None, None,
                        {"stopPrice": float(tp), "reduce_only": True, "triggerDirection": trig_up, "positionIdx": positionIdx})

async def close_market(symbol, side, positionIdx):
    pos = ex.fetch_positions([symbol])
    qty = 0.0
    for p in pos:
        if int(p.get("positionIdx",0)) == int(positionIdx):
            qty = abs(float(p.get("contracts",0) or 0))
            break
    if qty > 0:
        side_out = "sell" if side=="long" else "buy"
        ex.create_order(symbol, "market", side_out, qty, None, {"reduce_only": True, "positionIdx": positionIdx})

# ---------- STATE ----------
states = {
    sym: {
        "long_open": False, "short_open": False,
        "long_entry": None, "short_entry": None,
        "long_peak": None,  "short_trough": None,
        "day_start": None,  "day_disabled": False, "day_date": None
    } for sym in SYMBOLS
}

# ---------- TRADER PER SYMBOL ----------
async def trader_symbol(bot: Bot, symbol: str):
    # автоплечо на символ при старте (дополнительно делаем перед каждым входом)
    try:
        ex.set_leverage(LEVERAGE, symbol, params={"buyLeverage": LEVERAGE, "sellLeverage": LEVERAGE, "market":"linear"})
    except Exception:
        pass

    await bot.send_message(ADMIN_ID, f"Старт по {symbol} | hedge={'ON' if HEDGE_MODE else 'OFF'} | lev=x{LEVERAGE}")
    st = states[symbol]

    while True:
        try:
            # дневной старт/стоп
            now_date = datetime.now(timezone.utc).date()
            bal = 0.0
            try:
                bal = ex.fetch_balance().get("USDT",{}).get("free", 0.0)
            except Exception:
                pass

            if st["day_date"] != now_date:
                st["day_date"] = now_date
                st["day_start"] = bal
                st["day_disabled"] = False

            if DAILY_STOP_ENABLE and st["day_start"] is not None and bal > 0:
                dd = (st["day_start"] - bal) / max(st["day_start"], 1e-9)
                if dd >= DAILY_STOP_PCT and not st["day_disabled"]:
                    st["day_disabled"] = True
                    await bot.send_message(ADMIN_ID, f"{symbol}: Дневной стоп достигнут (просадка {dd*100:.2f}%). Входы до конца дня выключены.")

            # рыночные данные
            df = fetch_ohlcv(symbol)
            c = df["close"]
            emaf, emas = ema(c,EMAF), ema(c,EMAS)
            r = rsi(c,14)
            px = float(c.iloc[-1])

            bull = emaf.iloc[-2]<=emas.iloc[-2] and emaf.iloc[-1]>emas.iloc[-1] and r.iloc[-1]>50
            bear = emaf.iloc[-2]>=emas.iloc[-2] and emaf.iloc[-1]<emas.iloc[-1] and r.iloc[-1]<50
            near = any(abs(px-L)/px<0.0015 for L in gann_levels(px))

            # входы запрещены, если стоп-день включился
            can_enter = not st["day_disabled"]

            # ----- ВХОДЫ -----
            if can_enter:
                # LONG
                if bull and near and (HEDGE_MODE or not (st["long_open"] or st["short_open"])):
                    qty, sl = pos_size(bal, px, px*0.995, "long")
                    tp = px*1.005
                    await place(symbol, "long", qty, px, sl, tp, positionIdx=1 if HEDGE_MODE else 0)
                    st["long_open"]  = True
                    st["long_entry"] = px
                    st["long_peak"]  = px
                    await place_protective(symbol, "long", sl=sl, tp=tp, positionIdx=1 if HEDGE_MODE else 0)
                    await bot.send_message(ADMIN_ID, f"LONG {symbol} @ {px:.2f} | SL {sl:.2f} | TP {tp:.2f}")

                # SHORT
                if bear and near and (HEDGE_MODE or not (st["long_open"] or st["short_open"])):
                    qty, sl = pos_size(bal, px, px*1.005, "short")
                    tp = px*0.995
                    await place(symbol, "short", qty, px, sl, tp, positionIdx=2 if HEDGE_MODE else 0)
                    st["short_open"]   = True
                    st["short_entry"]  = px
                    st["short_trough"] = px
                    await place_protective(symbol, "short", sl=sl, tp=tp, positionIdx=2 if HEDGE_MODE else 0)
                    await bot.send_message(ADMIN_ID, f"SHORT {symbol} @ {px:.2f} | SL {sl:.2f} | TP {tp:.2f}")

            # ----- ВЫХОД ПО ОБРАТНОМУ СИГНАЛУ / ПЕРЕВОРОТ -----
            if EXIT_ON_OPPOSITE or REVERSE_ON_SIGNAL:
                # если открыт LONG и пришёл bear-сигнал
                if st["long_open"] and bear:
                    await close_market(symbol, "long", positionIdx=1 if HEDGE_MODE else 0)
                    st["long_open"] = False
                    st["long_entry"] = None
                    st["long_peak"] = None
                    if REVERSE_ON_SIGNAL and bear and near and can_enter:
                        qty, sl = pos_size(bal, px, px*1.005, "short")
                        tp = px*0.995
                        await place(symbol, "short", qty, px, sl, tp, positionIdx=2 if HEDGE_MODE else 0)
                        st["short_open"]   = True
                        st["short_entry"]  = px
                        st["short_trough"] = px
                        await place_protective(symbol, "short", sl=sl, tp=tp, positionIdx=2 if HEDGE_MODE else 0)

                # если открыт SHORT и пришёл bull-сигнал
                if st["short_open"] and bull:
                    await close_market(symbol, "short", positionIdx=2 if HEDGE_MODE else 0)
                    st["short_open"] = False
                    st["short_entry"] = None
                    st["short_trough"] = None
                    if REVERSE_ON_SIGNAL and bull and near and can_enter:
                        qty, sl = pos_size(bal, px, px*0.995, "long")
                        tp = px*1.005
                        await place(symbol, "long", qty, px, sl, tp, positionIdx=1 if HEDGE_MODE else 0)
                        st["long_open"]  = True
                        st["long_entry"] = px
                        st["long_peak"]  = px
                        await place_protective(symbol, "long", sl=sl, tp=tp, positionIdx=1 if HEDGE_MODE else 0)

            # ----- ТРЕЙЛИНГ и БЕЗУБЫТОК -----
            if st["long_open"] and st["long_entry"]:
                st["long_peak"] = max(st["long_peak"] or px, px)
                prof = pct(px, st["long_entry"])
                new_sl = None
                if BE_ENABLE and prof >= BE_TRIGGER_PCT:
                    new_sl = max(new_sl or -1e9, st["long_entry"])
                if TRAIL_ENABLE and prof >= TRAIL_ACTIVATE_PCT:
                    trail_sl = st["long_peak"] * (1 - TRAIL_DISTANCE_PCT)
                    new_sl = max(new_sl or -1e9, trail_sl)
                if new_sl:
                    await place_protective(symbol, "long", sl=new_sl, tp=None, positionIdx=1 if HEDGE_MODE else 0)

            if st["short_open"] and st["short_entry"]:
                st["short_trough"] = min(st["short_trough"] or px, px)
                prof = -pct(px, st["short_entry"])  # прибыль для шорта, когда px < entry
                new_sl = None
                if BE_ENABLE and prof >= BE_TRIGGER_PCT:
                    new_sl = min(new_sl or 1e9, st["short_entry"])
                if TRAIL_ENABLE and prof >= TRAIL_ACTIVATE_PCT:
                    trail_sl = st["short_trough"] * (1 + TRAIL_DISTANCE_PCT)
                    new_sl = min(new_sl or 1e9, trail_sl)
                if new_sl:
                    await place_protective(symbol, "short", sl=new_sl, tp=None, positionIdx=2 if HEDGE_MODE else 0)

            # ----- ОБНОВИМ ФЛАГИ ПО ФАКТУ -----
            try:
                pos = ex.fetch_positions([symbol])
                long_opened  = any((int(p.get("positionIdx",0))==1 or p.get("side")=="long")  and abs(float(p.get("contracts",0) or 0))>0 for p in pos)
                short_opened = any((int(p.get("positionIdx",0))==2 or p.get("side")=="short") and abs(float(p.get("contracts",0) or 0))>0 for p in pos)
            except Exception:
                long_opened = st["long_open"]
                short_opened = st["short_open"]

            if not long_opened:
                st["long_open"] = False
                st["long_entry"] = None
                st["long_peak"] = None
            if not short_opened:
                st["short_open"] = False
                st["short_entry"] = None
                st["short_trough"] = None

        except Exception as e:
            try:
                await bot.send_message(ADMIN_ID, f"{symbol}: ошибка: {e}")
            except Exception:
                pass
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
    await m.answer("Команды:\n/start_bot\n/stop_bot\n/status\n/symbols BTCUSDT,ETHUSDT\n/risk 5\n/flat")

_running = False
_tasks = []

@rt.message(Command("start_bot"))
async def start_bot(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    global _running, _tasks
    if _running: return await m.answer("Уже работает.")
    _running=True
    _tasks = [asyncio.create_task(trader_symbol(bot, s)) for s in SYMBOLS]
    await m.answer(f"Запускаю по: {', '.join(SYMBOLS)} | x{LEVERAGE}")

@rt.message(Command("stop_bot"))
async def stop_bot(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    global _running, _tasks
    _running=False
    for t in _tasks:
        t.cancel()
    _tasks=[]
    await m.answer("Остановлен (циклы завершатся через несколько секунд).")

@rt.message(Command("status"))
async def status(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    try:
        bal = ex.fetch_balance().get("USDT",{}).get("free", 0.0)
    except Exception:
        bal = 0.0
    hedge = "ON" if HEDGE_MODE else "OFF"
    await m.answer(f"SYMBOLS: {', '.join(SYMBOLS)}\nHedge: {hedge}\nLeverage: x{LEVERAGE}\nRisk: {int(RISK*100)}%\nBalance: {bal:.2f} USDT\nРаботает: {bool(_tasks)}")

@rt.message(F.text.startswith("/symbols"))
async def set_symbols(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    global SYMBOLS, states
    parts = m.text.split(maxsplit=1)
    if len(parts)>=2:
        SYMBOLS = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
        states = {
            sym: {
                "long_open": False, "short_open": False,
                "long_entry": None, "short_entry": None,
                "long_peak": None,  "short_trough": None,
                "day_start": None,  "day_disabled": False, "day_date": None
            } for sym in SYMBOLS
        }
        await m.answer(f"OK. SYMBOLS = {', '.join(SYMBOLS)}. Перезапусти /start_bot.")

@rt.message(F.text.startswith("/risk"))
async def set_risk(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    global RISK
    parts = m.text.split()
    if len(parts)>=2:
        RISK = float(parts[1])/100
        await m.answer(f"OK. Риск = {int(RISK*100)}%")

@rt.message(Command("flat"))
async def flat(m: Message):
    if m.from_user.id!=ADMIN_ID: return
    msgs = []
    for s in SYMBOLS:
        try: await close_market(s, "long",  positionIdx=1 if HEDGE_MODE else 0)
        except: pass
        try: await close_market(s, "short", positionIdx=2 if HEDGE_MODE else 0)
        except: pass
        states[s]["long_open"]=False; states[s]["short_open"]=False
        states[s]["long_entry"]=None; states[s]["short_entry"]=None
        states[s]["long_peak"]=None;  states[s]["short_trough"]=None
        msgs.append(s)
    await m.answer("Закрыл всё по рынку: " + ", ".join(msgs))

# ---------- FastAPI ----------
api = FastAPI()
@api.get("/health")
def health(): return {"ok": True}

@api.on_event("startup")
async def on_startup():
    asyncio.create_task(dp.start_polling(bot))

if __name__ == "__main__":
    uvicorn.run("app:api", host="0.0.0.0", port=int(os.getenv("PORT","8000")))
