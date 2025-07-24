import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import cv2
import time
import datetime
import pytz
import os
import winsound
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
#from python_telegram_bot import telegram

# === CONFIGURATION ===
SYMBOL = 'XAUUSD'
LOT = 0.01
POINT = 0.01
SPREAD_BUFFER = 30  # in pips (0.01 per pip for gold)
ATR_PERIOD = 20
MA_PERIOD = 100
RSI_PERIOD = 2
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 25
SL_ATR_MULT = 2.0
TP_ATR_MULT = 4.0
VOLUME_MULT = 3
CSV_LOG = 'gold_trades_log.csv'

# Nested trading configuration
MIN_ZONE_WIDTH = 5  # Minimum points for trading zone
MIN_TOUCHES = 3     # Minimum touches for valid trend line
M1_SCALP_TARGET = 10  # Points target for M1 scalping
M1_SCALP_STOP = 2     # Points stop for M1 scalping

# Telegram config
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'

class NestedTrendLineScalper:
    def __init__(self):
        self.m5_channels = []
        self.m1_nested_lines = []
        self.trading_zones = []
        self.active_trades = []
        
    # === MT5 CONNECTION ===
    def initialize_mt5(self):
        """Initialize MT5 with improved error handling"""
        possible_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Users\{}\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\terminal64.exe".format(os.getenv('USERNAME')),
            None
        ]
        
        print("Attempting to initialize MT5...")
        
        for path in possible_paths:
            print(f"Trying path: {path if path else 'Default MT5 path'}")
            
            if path:
                success = mt5.initialize(path=path)
            else:
                success = mt5.initialize()
                
            if success:
                print(f"✓ MT5 initialized successfully!")
                if self.attempt_login():
                    return True
                else:
                    print("✗ Login failed, but MT5 is initialized. Continuing with manual login...")
                    return self.manual_login_loop()
            else:
                error_code, error_msg = mt5.last_error()
                print(f"✗ Failed with error {error_code}: {error_msg}")
        
        print("\nAll automatic initialization attempts failed.")
        return self.manual_path_and_login()

    def attempt_login(self):
        """Attempt login with default credentials"""
        login = 210265375
        password = "Glo@1234567890"
        server = "Exness-MT5Trial9"
        
        print(f"Attempting login with account: {login}")
        authorized = mt5.login(login=login, password=password, server=server)
        
        if authorized:
            account_info = mt5.account_info()
            if account_info:
                print(f"✓ Successfully logged in!")
                print(f"Account: {account_info.login}, Balance: {account_info.balance}")
                return True
        return False

    def manual_login_loop(self):
        """Manual login loop for user input"""
        while True:
            print("\n--- Manual Login Required ---")
            try:
                login_input = input("Enter MT5 login/account number (or 'quit' to exit): ").strip()
                if login_input.lower() in ['quit', 'exit', '']:
                    print("Exiting...")
                    mt5.shutdown()
                    return False
                login = int(login_input)
            except ValueError:
                print("Invalid login number. Please enter a valid account number.")
                continue
                
            password = input("Enter MT5 password: ").strip()
            if not password:
                print("Password cannot be empty.")
                continue
                
            server = input("Enter MT5 server: ").strip()
            if not server:
                print("Server cannot be empty.")
                continue
            
            print(f"Attempting login with account: {login} on server: {server}")
            authorized = mt5.login(login=login, password=password, server=server)
            
            if authorized:
                account_info = mt5.account_info()
                if account_info:
                    print(f"✓ Successfully logged in!")
                    print(f"Account: {account_info.login}, Balance: {account_info.balance}")
                    return True
                else:
                    print("✗ Login appeared successful but cannot retrieve account info.")
            else:
                error_code, error_msg = mt5.last_error()
                print(f"✗ Login failed. Error {error_code}: {error_msg}")

    def manual_path_and_login(self):
        """Manual path input and login"""
        while True:
            path = input("\nEnter the full path to terminal64.exe (or 'quit' to exit): ").strip()
            if path.lower() in ['quit', 'exit', '']:
                print("Exiting...")
                return False
                
            path = path.strip('"').strip("'")
            
            if not os.path.exists(path):
                print(f"✗ File not found: {path}")
                continue
                
            print(f"Trying to initialize with: {path}")
            success = mt5.initialize(path=path)
            
            if success:
                print("✓ MT5 initialized successfully!")
                if self.manual_login_loop():
                    return True
            else:
                error_code, error_msg = mt5.last_error()
                print(f"✗ Initialization failed. Error {error_code}: {error_msg}")

    # === DATA FETCHING ===
    def get_rates(self, symbol, timeframe, n):
        """Get rates with error handling"""
        try:
            utc_from = datetime.datetime.now(pytz.utc) - datetime.timedelta(minutes=n*2)
            rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n)
            
            if rates is None or len(rates) == 0:
                print(f"Warning: No rates data received for {symbol}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching rates: {e}")
            return None

    # === TREND LINE DETECTION ===
    def find_pivot_points(self, df, window=5):
        """Find pivot highs and lows"""
        if df is None or len(df) < window * 2:
            return [], []
        
        # Find local maxima (resistance pivots)
        highs_idx = argrelextrema(df['high'].values, np.greater, order=window)[0]
        resistance_pivots = [(df.index[i], df['high'].iloc[i]) for i in highs_idx]
        
        # Find local minima (support pivots)
        lows_idx = argrelextrema(df['low'].values, np.less, order=window)[0]
        support_pivots = [(df.index[i], df['low'].iloc[i]) for i in lows_idx]
        
        return support_pivots, resistance_pivots

    def calculate_trend_line(self, pivots):
        """Calculate trend line from pivot points"""
        if len(pivots) < 2:
            return None
        
        # Convert timestamps to numeric values for regression
        times = [pivot[0].timestamp() for pivot in pivots]
        prices = [pivot[1] for pivot in pivots]
        
        # Use linear regression to find best fit line
        X = np.array(times).reshape(-1, 1)
        y = np.array(prices)
        
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        r_squared = reg.score(X, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'pivots': pivots,
            'touches': len(pivots)
        }

    def validate_trend_line(self, trend_line, df, tolerance=2):
        """Validate trend line by checking how many points it touches"""
        if trend_line is None:
            return False
        
        touches = 0
        slope = trend_line['slope']
        intercept = trend_line['intercept']
        
        for idx, row in df.iterrows():
            timestamp = idx.timestamp()
            expected_price = slope * timestamp + intercept
            
            # Check if price touches the line within tolerance
            if abs(row['high'] - expected_price) <= tolerance * POINT:
                touches += 1
            elif abs(row['low'] - expected_price) <= tolerance * POINT:
                touches += 1
        
        return touches >= MIN_TOUCHES

    def detect_m5_channels(self, symbol):
        """Detect M5 major trend channels"""
        print("Detecting M5 channels...")
        m5_data = self.get_rates(symbol, mt5.TIMEFRAME_M5, 200)
        
        if m5_data is None:
            return []
        
        support_pivots, resistance_pivots = self.find_pivot_points(m5_data, window=10)
        
        channels = []
        
        # Find support trend lines
        if len(support_pivots) >= 3:
            for i in range(len(support_pivots) - 2):
                for j in range(i + 2, len(support_pivots)):
                    pivot_subset = support_pivots[i:j+1]
                    trend_line = self.calculate_trend_line(pivot_subset)
                    
                    if trend_line and self.validate_trend_line(trend_line, m5_data):
                        # Find corresponding resistance
                        resistance_line = self.find_parallel_resistance(resistance_pivots, trend_line, m5_data)
                        
                        if resistance_line:
                            channels.append({
                                'support': trend_line,
                                'resistance': resistance_line,
                                'timeframe': 'M5',
                                'strength': min(trend_line['touches'], resistance_line['touches'])
                            })
        
        # Sort by strength and return top channels
        channels.sort(key=lambda x: x['strength'], reverse=True)
        print(f"Found {len(channels)} M5 channels")
        return channels[:3]  # Return top 3 channels

    def find_parallel_resistance(self, resistance_pivots, support_line, df):
        """Find resistance line parallel to support"""
        if len(resistance_pivots) < 2:
            return None
        
        support_slope = support_line['slope']
        
        # Look for resistance pivots that could form parallel line
        for i in range(len(resistance_pivots) - 1):
            for j in range(i + 1, len(resistance_pivots)):
                pivot_pair = [resistance_pivots[i], resistance_pivots[j]]
                trend_line = self.calculate_trend_line(pivot_pair)
                
                if trend_line and abs(trend_line['slope'] - support_slope) < support_slope * 0.1:  # Similar slope
                    if self.validate_trend_line(trend_line, df):
                        return trend_line
        
        return None

    # TODO: Implement nested trend line detection - M1 lines within M5 boundaries
    def detect_nested_trend_lines(self, symbol):
        """Find M1 support/resistance lines that exist INSIDE M5 channel"""
        print("Detecting nested M1 trend lines...")
        
        # Get M5 major channels
        m5_channels = self.detect_m5_channels(symbol)
        if not m5_channels:
            print("No M5 channels found")
            return []
        
        # Get M1 data for detailed analysis
        m1_data = self.get_rates(symbol, mt5.TIMEFRAME_M1, 500)
        if m1_data is None:
            return []
        
        nested_structures = []
        
        for channel in m5_channels:
            print(f"Analyzing M1 lines within M5 channel (strength: {channel['strength']})")
            
            # Find M1 pivot points
            m1_support_pivots, m1_resistance_pivots = self.find_pivot_points(m1_data, window=3)
            
            # Filter pivots that are within M5 channel boundaries
            channel_support_pivots = []
            channel_resistance_pivots = []
            
            current_time = datetime.datetime.now().timestamp()
            
            for pivot in m1_support_pivots:
                pivot_time = pivot[0].timestamp()
                pivot_price = pivot[1]
                
                # Calculate M5 support/resistance at this time
                m5_support_price = channel['support']['slope'] * pivot_time + channel['support']['intercept']
                m5_resistance_price = channel['resistance']['slope'] * pivot_time + channel['resistance']['intercept']
                
                # Check if M1 pivot is within M5 boundaries
                if m5_support_price <= pivot_price <= m5_resistance_price:
                    channel_support_pivots.append(pivot)
            
            for pivot in m1_resistance_pivots:
                pivot_time = pivot[0].timestamp()
                pivot_price = pivot[1]
                
                m5_support_price = channel['support']['slope'] * pivot_time + channel['support']['intercept']
                m5_resistance_price = channel['resistance']['slope'] * pivot_time + channel['resistance']['intercept']
                
                if m5_support_price <= pivot_price <= m5_resistance_price:
                    channel_resistance_pivots.append(pivot)
            
            # Create M1 trend lines from filtered pivots
            m1_lines = []
            
            # Process support lines
            if len(channel_support_pivots) >= 2:
                for i in range(len(channel_support_pivots) - 1):
                    for j in range(i + 1, len(channel_support_pivots)):
                        pivot_pair = [channel_support_pivots[i], channel_support_pivots[j]]
                        trend_line = self.calculate_trend_line(pivot_pair)
                        
                        if trend_line and trend_line['r_squared'] > 0.7:
                            trend_line['type'] = 'support'
                            m1_lines.append(trend_line)
            
            # Process resistance lines
            if len(channel_resistance_pivots) >= 2:
                for i in range(len(channel_resistance_pivots) - 1):
                    for j in range(i + 1, len(channel_resistance_pivots)):
                        pivot_pair = [channel_resistance_pivots[i], channel_resistance_pivots[j]]
                        trend_line = self.calculate_trend_line(pivot_pair)
                        
                        if trend_line and trend_line['r_squared'] > 0.7:
                            trend_line['type'] = 'resistance'
                            m1_lines.append(trend_line)
            
            if m1_lines:
                nested_structures.append({
                    'm5_channel': channel,
                    'm1_lines': m1_lines,
                    'nested_count': len(m1_lines)
                })
                print(f"Found {len(m1_lines)} nested M1 lines in this M5 channel")
        
        print(f"Total nested structures: {len(nested_structures)}")
        return nested_structures

    # TODO: Create specific trading zones between adjacent M1 trend lines
    def create_m1_trading_zones(self, m5_structure, m1_nested_lines):
        """Create trading zones between adjacent M1 levels within M5 boundaries"""
        print("Creating M1 trading zones...")
        
        if not m1_nested_lines:
            return []
        
        current_time = datetime.datetime.now().timestamp()
        trading_zones = []
        
        # Get current prices for all M1 lines
        m1_levels = []
        for line in m1_nested_lines:
            current_price = line['slope'] * current_time + line['intercept']
            m1_levels.append({
                'price': current_price,
                'type': line['type'],
                'line': line
            })
        
        # Sort levels by price
        m1_levels.sort(key=lambda x: x['price'])
        
        # Create zones between adjacent levels
        for i in range(len(m1_levels) - 1):
            lower_level = m1_levels[i]
            upper_level = m1_levels[i + 1]
            
            zone_width = upper_level['price'] - lower_level['price']
            
            # Validate zone has minimum width
            if zone_width >= MIN_ZONE_WIDTH * POINT:
                # Get M5 boundaries at current time
                m5_support = m5_structure['support']['slope'] * current_time + m5_structure['support']['intercept']
                m5_resistance = m5_structure['resistance']['slope'] * current_time + m5_structure['resistance']['intercept']
                
                # Ensure zone is within M5 boundaries
                if m5_support <= lower_level['price'] and upper_level['price'] <= m5_resistance:
                    zone = {
                        'lower_bound': lower_level['price'],
                        'upper_bound': upper_level['price'],
                        'width': zone_width,
                        'lower_line': lower_level['line'],
                        'upper_line': upper_level['line'],
                        'zone_id': f"Zone_{i}_{int(current_time)}",
                        'm5_support': m5_support,
                        'm5_resistance': m5_resistance,
                        'scalp_target': min(zone_width * 0.8, M1_SCALP_TARGET * POINT),
                        'scalp_stop': M1_SCALP_STOP * POINT
                    }
                    trading_zones.append(zone)
        
        print(f"Created {len(trading_zones)} valid trading zones")
        return trading_zones

    # TODO: Execute M1 scalping within nested zones respecting M5 boundaries
    def execute_nested_m1_scalping(self, symbol, nested_structure, trend_alignment):
        """Execute high-frequency M1 scalping within nested zones"""
        
        m5_channel = nested_structure['m5_channel']
        m1_lines = nested_structure['m1_lines']
        
        # Create trading zones
        trading_zones = self.create_m1_trading_zones(m5_channel, m1_lines)
        
        if not trading_zones:
            print("No valid trading zones found")
            return False
        
        # Get current market data
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        current_price = (tick.ask + tick.bid) / 2
        current_time = datetime.datetime.now().timestamp()
        
        # Check M5 trend alignment for trade direction bias
        m5_support_current = m5_channel['support']['slope'] * current_time + m5_channel['support']['intercept']
        m5_resistance_current = m5_channel['resistance']['slope'] * current_time + m5_channel['resistance']['intercept']
        m5_mid = (m5_support_current + m5_resistance_current) / 2
        
        # Determine trend bias
        if current_price > m5_mid:
            trend_bias = 'bullish'  # Prefer long trades
        else:
            trend_bias = 'bearish'  # Prefer short trades
        
        trades_executed = 0
        
        # Execute scalping in each zone
        for zone in trading_zones:
            # Check if current price is in this zone
            if zone['lower_bound'] <= current_price <= zone['upper_bound']:
                print(f"Price in zone: {zone['lower_bound']:.2f} - {zone['upper_bound']:.2f}")
                
                # Scalping logic based on zone position and trend bias
                zone_position = (current_price - zone['lower_bound']) / zone['width']
                
                # BUY at lower part of zone (if bullish bias or price near support)
                if (zone_position < 0.3 and trend_bias == 'bullish') or zone_position < 0.1:
                    if self.execute_scalp_trade(symbol, 'BUY', zone, tick.ask):
                        trades_executed += 1
                
                # SELL at upper part of zone (if bearish bias or price near resistance)
                elif (zone_position > 0.7 and trend_bias == 'bearish') or zone_position > 0.9:
                    if self.execute_scalp_trade(symbol, 'SELL', zone, tick.bid):
                        trades_executed += 1
                
                # Mean reversion trades
                elif 0.4 <= zone_position <= 0.6:
                    # Price in middle of zone - look for reversal signals
                    m1_data = self.get_rates(symbol, mt5.TIMEFRAME_M1, 10)
                    if m1_data is not None and len(m1_data) >= 3:
                        # Simple momentum check
                        recent_change = m1_data['close'].iloc[-1] - m1_data['close'].iloc[-3]
                        
                        if recent_change > 0 and zone_position > 0.5:  # Rising into upper zone
                            if self.execute_scalp_trade(symbol, 'SELL', zone, tick.bid):
                                trades_executed += 1
                        elif recent_change < 0 and zone_position < 0.5:  # Falling into lower zone
                            if self.execute_scalp_trade(symbol, 'BUY', zone, tick.ask):
                                trades_executed += 1
        
        print(f"Executed {trades_executed} scalp trades")
        return trades_executed > 0

    def execute_scalp_trade(self, symbol, direction, zone, entry_price):
        """Execute individual scalp trade with tight stops"""
        
        # Calculate stop loss and take profit
        if direction == 'BUY':
            stop_loss = entry_price - zone['scalp_stop']
            take_profit = entry_price + zone['scalp_target']
            # Ensure we don't exceed zone boundaries
            take_profit = min(take_profit, zone['upper_bound'] - POINT)
            stop_loss = max(stop_loss, zone['m5_support'])  # Never go below M5 support
            
            order_type = mt5.ORDER_TYPE_BUY
            price = entry_price
            
        else:  # SELL
            stop_loss = entry_price + zone['scalp_stop']
            take_profit = entry_price - zone['scalp_target']
            # Ensure we don't exceed zone boundaries
            take_profit = max(take_profit, zone['lower_bound'] + POINT)
            stop_loss = min(stop_loss, zone['m5_resistance'])  # Never go above M5 resistance
            
            order_type = mt5.ORDER_TYPE_SELL
            price = entry_price
        
        # Risk validation
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.5:  # Minimum 1.5:1 reward/risk
            print(f"Trade rejected: Poor R/R ratio {reward/risk:.2f}")
            return False
        
        # Place the order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": LOT,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,  # Tighter deviation for scalping
            "magic": 123457,  # Different magic for nested scalping
            "comment": f"NestedScalp_{direction}_{zone['zone_id']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✓ {direction} scalp executed at {price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            
            # Log the trade
            self.log_trade({
                'time': datetime.datetime.now(),
                'type': f'nested_scalp_{direction.lower()}',
                'price': price,
                'sl': stop_loss,
                'tp': take_profit,
                'zone_id': zone['zone_id'],
                'risk': risk,
                'reward': reward,
                'rr_ratio': reward/risk
            })
            
            # Sound alert
            self.play_sound()
            return True
        else:
            print(f"✗ {direction} scalp failed: {result.retcode} - {result.comment}")
            return False

    # === UTILITY FUNCTIONS ===
    def play_sound(self):
        """Play sound alert"""
        try:
            winsound.Beep(1000, 200)  # Shorter beep for scalping
        except Exception as e:
            print(f"Sound alert failed: {e}")

    def log_trade(self, data):
        """Log trade to CSV"""
        try:
            df = pd.DataFrame([data])
            if not os.path.exists(CSV_LOG):
                df.to_csv(CSV_LOG, index=False)
            else:
                df.to_csv(CSV_LOG, mode='a', header=False, index=False)
        except Exception as e:
            print(f"CSV logging failed: {e}")

    def is_trading_time(self):
        """Check if it's appropriate trading time"""
        try:
            now = datetime.datetime.now(pytz.timezone('Europe/London'))
            # Avoid NY/London overlap and low volatility times
            if 13 <= now.hour < 17:  # NY/London overlap
                return False
            if 22 <= now.hour or now.hour < 6:  # Asian session low volatility
                return False
            return True
        except Exception as e:
            print(f"Time filter error: {e}")
            return True

    # === MAIN LOOP ===
    def run(self):
        """Main trading loop"""
        if not self.initialize_mt5():
            print("Failed to initialize MT5. Exiting.")
            return
        
        print("Starting Nested Trend Line Scalping Bot...")
        print(f"Target: 10-20 scalp trades per day with 75%+ win rate")
        
        last_analysis_time = 0
        analysis_interval = 300  # Reanalyze nested structure every 5 minutes
        
        while True:
            try:
                current_time = time.time()
                
                # Check if it's trading time
                if not self.is_trading_time():
                    print("Outside trading hours. Waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Reanalyze nested structure periodically
                if current_time - last_analysis_time > analysis_interval:
                    print("\n=== Analyzing Nested Trend Structure ===")
                    nested_structures = self.detect_nested_trend_lines(SYMBOL)
                    last_analysis_time = current_time
                    
                    if not nested_structures:
                        print("No nested structures found. Waiting...")
                        time.sleep(60)
                        continue
                else:
                    # Use existing structures if available
                    if not hasattr(self, 'nested_structures') or not self.nested_structures:
                        time.sleep(10)
                        continue
                    nested_structures = self.nested_structures
                
                # Store for next iteration
                self.nested_structures = nested_structures
                
                # Execute scalping on each nested structure
                total_trades = 0
                for structure in nested_structures:
                    if structure['nested_count'] >= 3:  # Need at least 3 M1 lines
                        print(f"\n--- Executing scalping in structure with {structure['nested_count']} M1 lines ---")
                        
                        # Simple trend alignment check
                        m1_recent = self.get_rates(SYMBOL, mt5.TIMEFRAME_M1, 20)
                        trend_alignment = 'neutral'
                        
                        if m1_recent is not None and len(m1_recent) >= 10:
                            ma_short = m1_recent['close'].rolling(5).mean().iloc[-1]
                            ma_long = m1_recent['close'].rolling(10).mean().iloc[-1]
                            
                            if ma_short > ma_long:
                                trend_alignment = 'bullish'
                            elif ma_short < ma_long:
                                trend_alignment = 'bearish'
                        
                        # Execute nested scalping
                        if self.execute_nested_m1_scalping(SYMBOL, structure, trend_alignment):
                            total_trades += 1
                
                if total_trades == 0:
                    print("No scalping opportunities found in current structures")
                
                # Short sleep for high-frequency scalping
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(30)
        
        # Cleanup
        mt5.shutdown()
        print("Nested Trend Line Scalping Bot stopped.")

if __name__ == "__main__":
    scalper = NestedTrendLineScalper()
    scalper.run()