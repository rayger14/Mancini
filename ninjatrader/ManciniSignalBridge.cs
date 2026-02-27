// ManciniSignalBridge — NinjaScript Strategy for NinjaTrader 8
//
// Bridges Python strategy engine to NinjaTrader execution:
//   - Writes 1-min bar data as JSON files for Python to read
//   - Polls for signal files from Python (enter_long, update_stop, flatten)
//   - Executes bracket orders (EnterLong + SetStopLoss + SetProfitTarget)
//   - Writes fill confirmations and position state back to Python
//
// Installation:
//   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Strategies\
//   2. In NinjaTrader: Tools > NinjaScript Editor > right-click > Compile
//   3. Apply to a MES 1-min chart
//
// Shared directory: C:\ManciniShared\ (configurable via SharedDirectory parameter)

#region Using declarations
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ManciniSignalBridge : Strategy
    {
        #region Parameters
        private string SharedDirectory = @"C:\ManciniShared";
        private int HistoryBarsToWrite = 400;
        private int HeartbeatIntervalMs = 5000;
        private int SignalPollIntervalMs = 500;
        private int MaxSignalAgeSec = 120;  // reject signals older than 2 min
        #endregion

        #region Internal State
        private int barNumber = 0;
        private bool sessionInitialized = false;
        private string currentSignalId = "";
        private int fillCounter = 0;
        private Timer heartbeatTimer;
        private Timer signalPollTimer;
        private string barsDir;
        private string signalsDir;
        private string fillsDir;
        private string stateDir;
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Bridges Python Mancini strategy to NinjaTrader execution";
                Name = "ManciniSignalBridge";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 300;  // flatten 5 min before close
                IsFillLimitOnTouch = false;
                TraceOrders = true;
                BarsRequiredToTrade = 20;
                IsInstantiatedOnEachOptimizationIteration = false;
            }
            else if (State == State.Configure)
            {
                // Primary data series is already the chart's 1-min bars
            }
            else if (State == State.DataLoaded)
            {
                // Create directory structure
                barsDir = Path.Combine(SharedDirectory, "bars");
                signalsDir = Path.Combine(SharedDirectory, "signals");
                fillsDir = Path.Combine(SharedDirectory, "fills");
                stateDir = Path.Combine(SharedDirectory, "state");

                Directory.CreateDirectory(barsDir);
                Directory.CreateDirectory(signalsDir);
                Directory.CreateDirectory(fillsDir);
                Directory.CreateDirectory(stateDir);

                Print("ManciniSignalBridge: Directories ready at " + SharedDirectory);
            }
            else if (State == State.Realtime)
            {
                // Start timers only in real-time mode
                heartbeatTimer = new Timer(OnHeartbeatTimer, null, 0, HeartbeatIntervalMs);
                signalPollTimer = new Timer(OnSignalPollTimer, null, 1000, SignalPollIntervalMs);
                Print("ManciniSignalBridge: Real-time mode — timers started");
            }
            else if (State == State.Terminated)
            {
                if (heartbeatTimer != null) heartbeatTimer.Dispose();
                if (signalPollTimer != null) signalPollTimer.Dispose();
            }
        }

        protected override void OnBarUpdate()
        {
            // Only process the primary series
            if (BarsInProgress != 0) return;

            // Only process RTH bars
            if (Bars.IsFirstBarOfSession && !sessionInitialized)
            {
                OnSessionStart();
                sessionInitialized = true;
            }

            // Write bar data for Python
            WriteBarFile();
            barNumber++;

            // Write current position state
            WritePositionState();

            // Also poll signals on bar close (in addition to timer)
            if (State == State.Realtime)
            {
                PollSignals();
            }
        }

        protected override void OnExecutionUpdate(Execution execution,
            string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Write fill confirmation for Python
            fillCounter++;
            WriteFillFile(execution, price, quantity, time);
            WritePositionState();
        }

        protected override void OnOrderUpdate(Order order, double limitPrice,
            double stopPrice, int quantity, int filled, double averageFillPrice,
            OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            if (orderState == OrderState.Rejected)
            {
                Print("ORDER REJECTED: " + comment);
                WriteRejectionFile(order, comment);
            }
        }

        #region Session Management

        private void OnSessionStart()
        {
            barNumber = 0;
            fillCounter = 0;
            currentSignalId = "";

            // Write history file with prior bars for Python
            WriteHistoryFile();
            Print("ManciniSignalBridge: Session started, history written");
        }

        #endregion

        #region Bar Writing

        private void WriteBarFile()
        {
            string dateStr = Time[0].ToString("yyyyMMdd");
            string timeStr = Time[0].ToString("HHmm");
            string filename = string.Format("bar_{0}_{1}.json", dateStr, timeStr);
            string path = Path.Combine(barsDir, filename);

            string json = string.Format(
                "{{\n" +
                "  \"timestamp\": \"{0}\",\n" +
                "  \"open\": {1},\n" +
                "  \"high\": {2},\n" +
                "  \"low\": {3},\n" +
                "  \"close\": {4},\n" +
                "  \"volume\": {5},\n" +
                "  \"bar_number\": {6},\n" +
                "  \"instrument\": \"{7}\"\n" +
                "}}",
                Time[0].ToString("yyyy-MM-ddTHH:mm:sszzz"),
                Open[0], High[0], Low[0], Close[0],
                (long)Volume[0], barNumber, Instrument.FullName);

            AtomicWrite(path, json);
        }

        private void WriteHistoryFile()
        {
            string dateStr = Time[0].ToString("yyyyMMdd");
            string filename = string.Format("history_{0}.json", dateStr);
            string path = Path.Combine(barsDir, filename);

            // Collect prior day bars
            var priorBars = new List<string>();
            var currentBars = new List<string>();

            int totalBars = Math.Min(CurrentBar + 1, HistoryBarsToWrite);
            DateTime sessionDate = Time[0].Date;

            for (int i = totalBars - 1; i >= 0; i--)
            {
                string barJson = string.Format(
                    "    {{\n" +
                    "      \"timestamp\": \"{0}\",\n" +
                    "      \"open\": {1},\n" +
                    "      \"high\": {2},\n" +
                    "      \"low\": {3},\n" +
                    "      \"close\": {4},\n" +
                    "      \"volume\": {5}\n" +
                    "    }}",
                    Time[i].ToString("yyyy-MM-ddTHH:mm:sszzz"),
                    Open[i], High[i], Low[i], Close[i], (long)Volume[i]);

                if (Time[i].Date < sessionDate)
                    priorBars.Add(barJson);
                else
                    currentBars.Add(barJson);
            }

            string json = string.Format(
                "{{\n" +
                "  \"session_date\": \"{0}\",\n" +
                "  \"instrument\": \"{1}\",\n" +
                "  \"prior_day_bars\": [\n{2}\n  ],\n" +
                "  \"current_day_bars\": [\n{3}\n  ]\n" +
                "}}",
                sessionDate.ToString("yyyy-MM-dd"),
                Instrument.FullName,
                string.Join(",\n", priorBars),
                string.Join(",\n", currentBars));

            AtomicWrite(path, json);
            Print(string.Format("History written: {0} prior bars, {1} current bars",
                priorBars.Count, currentBars.Count));
        }

        #endregion

        #region Signal Reading

        private void PollSignals()
        {
            if (!Directory.Exists(signalsDir)) return;

            var signalFiles = Directory.GetFiles(signalsDir, "signal_*.json")
                .OrderBy(f => f).ToArray();

            foreach (string filePath in signalFiles)
            {
                try
                {
                    string content = File.ReadAllText(filePath);

                    // Simple JSON parsing (avoid external dependencies)
                    if (content.Contains("\"status\": \"UNREAD\""))
                    {
                        string action = ExtractJsonValue(content, "action");
                        string signalId = ExtractJsonValue(content, "signal_id");
                        string timestamp = ExtractJsonValue(content, "timestamp");

                        // Check signal age
                        DateTime signalTime;
                        if (DateTime.TryParse(timestamp, out signalTime))
                        {
                            double ageSec = (DateTime.Now - signalTime).TotalSeconds;
                            if (ageSec > MaxSignalAgeSec)
                            {
                                Print("Rejecting stale signal: " + signalId);
                                UpdateSignalStatus(filePath, content, "REJECTED");
                                continue;
                            }
                        }

                        // Dispatch by action type
                        switch (action)
                        {
                            case "enter_long":
                                ExecuteEntrySignal(content, signalId);
                                break;
                            case "update_stop":
                                ExecuteUpdateStop(content, signalId);
                                break;
                            case "flatten":
                                ExecuteFlatten(content, signalId);
                                break;
                            case "partial_exit":
                                ExecutePartialExit(content, signalId);
                                break;
                            default:
                                Print("Unknown signal action: " + action);
                                break;
                        }

                        // Mark as read
                        UpdateSignalStatus(filePath, content, "EXECUTED");
                    }
                }
                catch (Exception ex)
                {
                    Print("Error reading signal file: " + ex.Message);
                }
            }
        }

        private void ExecuteEntrySignal(string content, string signalId)
        {
            // Don't enter if already in a position
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                Print("Ignoring entry signal — already in position");
                return;
            }

            int quantity = ParseInt(ExtractJsonValue(content, "quantity"), 1);
            double stopPrice = ParseDouble(ExtractJsonValue(content, "stop_price"), 0);
            double targetPrice = ParseDouble(ExtractJsonValue(content, "target_price"), 0);
            string signalType = ExtractJsonValue(content, "signal_type");

            currentSignalId = signalId;

            // Submit bracket order
            SetStopLoss("ManciniEntry", CalculationMode.Price, stopPrice, false);
            SetProfitTarget("ManciniEntry", CalculationMode.Price, targetPrice);
            EnterLong(quantity, "ManciniEntry");

            Print(string.Format("ENTRY: {0} @ market, stop={1:F2}, target={2:F2} [{3}]",
                quantity, stopPrice, targetPrice, signalType));
        }

        private void ExecuteUpdateStop(string content, string signalId)
        {
            double newStop = ParseDouble(ExtractJsonValue(content, "new_stop_price"), 0);
            string reason = ExtractJsonValue(content, "reason");

            if (Position.MarketPosition == MarketPosition.Long && newStop > 0)
            {
                SetStopLoss("ManciniEntry", CalculationMode.Price, newStop, false);
                Print(string.Format("STOP UPDATE: {0:F2} — {1}", newStop, reason));
            }
        }

        private void ExecuteFlatten(string content, string signalId)
        {
            string reason = ExtractJsonValue(content, "reason");

            if (Position.MarketPosition != MarketPosition.Flat)
            {
                ExitLong("ManciniEntry");
                Print("FLATTEN: " + reason);
            }
        }

        private void ExecutePartialExit(string content, string signalId)
        {
            int quantity = ParseInt(ExtractJsonValue(content, "quantity"), 0);
            double newStop = ParseDouble(ExtractJsonValue(content, "new_stop_price"), 0);
            string reason = ExtractJsonValue(content, "reason");

            if (Position.MarketPosition == MarketPosition.Long && quantity > 0)
            {
                ExitLong(quantity, "PartialExit", "ManciniEntry");
                if (newStop > 0)
                {
                    SetStopLoss("ManciniEntry", CalculationMode.Price, newStop, false);
                }
                Print(string.Format("PARTIAL EXIT: {0} contracts, new stop={1:F2} — {2}",
                    quantity, newStop, reason));
            }
        }

        #endregion

        #region Fill Writing

        private void WriteFillFile(Execution execution, double price, int quantity, DateTime time)
        {
            string dateStr = time.ToString("yyyyMMdd");
            string filename = string.Format("fill_{0}_{1:D3}.json", dateStr, fillCounter);
            string path = Path.Combine(fillsDir, filename);

            string action = execution.Order.IsLong ? "entry_fill" : "exit_fill";
            double commission = execution.Commission;

            string json = string.Format(
                "{{\n" +
                "  \"fill_id\": \"fill_{0}_{1:D3}\",\n" +
                "  \"signal_id\": \"{2}\",\n" +
                "  \"action\": \"{3}\",\n" +
                "  \"instrument\": \"{4}\",\n" +
                "  \"price\": {5},\n" +
                "  \"quantity\": {6},\n" +
                "  \"timestamp\": \"{7}\",\n" +
                "  \"commission\": {8},\n" +
                "  \"order_id\": \"{9}\"\n" +
                "}}",
                dateStr, fillCounter,
                currentSignalId,
                action,
                Instrument.FullName,
                price, quantity,
                time.ToString("yyyy-MM-ddTHH:mm:sszzz"),
                commission,
                execution.Order.Id);

            AtomicWrite(path, json);
            Print(string.Format("Fill written: {0} {1} @ {2:F2}", action, quantity, price));
        }

        private void WriteRejectionFile(Order order, string reason)
        {
            fillCounter++;
            string dateStr = DateTime.Now.ToString("yyyyMMdd");
            string filename = string.Format("fill_{0}_{1:D3}.json", dateStr, fillCounter);
            string path = Path.Combine(fillsDir, filename);

            string json = string.Format(
                "{{\n" +
                "  \"fill_id\": \"fill_{0}_{1:D3}\",\n" +
                "  \"signal_id\": \"{2}\",\n" +
                "  \"action\": \"rejected\",\n" +
                "  \"instrument\": \"{3}\",\n" +
                "  \"reason\": \"{4}\",\n" +
                "  \"timestamp\": \"{5}\"\n" +
                "}}",
                dateStr, fillCounter,
                currentSignalId,
                Instrument.FullName,
                reason.Replace("\"", "'"),
                DateTime.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"));

            AtomicWrite(path, json);
        }

        #endregion

        #region Position State

        private void WritePositionState()
        {
            string path = Path.Combine(stateDir, "position.json");

            string marketPos = "flat";
            double avgEntry = 0;
            int qty = 0;
            double unrealizedPnl = 0;
            double workingStop = 0;
            double workingTarget = 0;

            if (Position.MarketPosition == MarketPosition.Long)
            {
                marketPos = "long";
                avgEntry = Position.AveragePrice;
                qty = Position.Quantity;
                unrealizedPnl = Position.GetUnrealizedProfitLoss(
                    PerformanceUnit.Points, Close[0]);
            }

            // Find working stop/target orders
            foreach (Order order in Orders)
            {
                if (order.OrderState == OrderState.Working || order.OrderState == OrderState.Accepted)
                {
                    if (order.OrderType == OrderType.StopMarket)
                        workingStop = order.StopPrice;
                    else if (order.OrderType == OrderType.Limit)
                        workingTarget = order.LimitPrice;
                }
            }

            string json = string.Format(
                "{{\n" +
                "  \"timestamp\": \"{0}\",\n" +
                "  \"market_position\": \"{1}\",\n" +
                "  \"quantity\": {2},\n" +
                "  \"avg_entry_price\": {3},\n" +
                "  \"unrealized_pnl\": {4:F2},\n" +
                "  \"working_stop\": {5},\n" +
                "  \"working_target\": {6}\n" +
                "}}",
                DateTime.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"),
                marketPos, qty, avgEntry,
                unrealizedPnl, workingStop, workingTarget);

            AtomicWrite(path, json);
        }

        #endregion

        #region Heartbeat

        private void OnHeartbeatTimer(object state)
        {
            try
            {
                string path = Path.Combine(stateDir, "nt_heartbeat.json");
                string json = string.Format(
                    "{{\n" +
                    "  \"timestamp\": \"{0}\",\n" +
                    "  \"status\": \"running\",\n" +
                    "  \"bars_processed\": {1},\n" +
                    "  \"session_date\": \"{2}\"\n" +
                    "}}",
                    DateTime.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"),
                    barNumber,
                    DateTime.Now.ToString("yyyy-MM-dd"));

                AtomicWrite(path, json);

                // Also check Python heartbeat
                CheckPythonHeartbeat();
            }
            catch { }
        }

        private void OnSignalPollTimer(object state)
        {
            try
            {
                PollSignals();
            }
            catch { }
        }

        private void CheckPythonHeartbeat()
        {
            string path = Path.Combine(stateDir, "py_heartbeat.json");
            if (!File.Exists(path)) return;

            try
            {
                string content = File.ReadAllText(path);
                string timestamp = ExtractJsonValue(content, "timestamp");
                DateTime pyTime;
                if (DateTime.TryParse(timestamp, out pyTime))
                {
                    double ageSec = (DateTime.Now - pyTime).TotalSeconds;
                    if (ageSec > 30)
                    {
                        Print("WARNING: Python heartbeat stale (" + ageSec.ToString("F0") + "s)");
                    }
                }
            }
            catch { }
        }

        #endregion

        #region Utilities

        private void AtomicWrite(string path, string content)
        {
            string tmpPath = path + ".tmp";
            File.WriteAllText(tmpPath, content);
            if (File.Exists(path)) File.Delete(path);
            File.Move(tmpPath, path);
        }

        private void UpdateSignalStatus(string filePath, string content, string newStatus)
        {
            string updated = content.Replace("\"UNREAD\"", "\"" + newStatus + "\"");
            AtomicWrite(filePath, updated);
        }

        /// <summary>
        /// Simple JSON value extraction (avoids needing Newtonsoft).
        /// Handles: "key": "value" and "key": 123
        /// </summary>
        private string ExtractJsonValue(string json, string key)
        {
            string pattern = "\"" + key + "\":";
            int idx = json.IndexOf(pattern);
            if (idx < 0) return "";

            int valueStart = idx + pattern.Length;
            // Skip whitespace
            while (valueStart < json.Length && json[valueStart] == ' ') valueStart++;

            if (valueStart >= json.Length) return "";

            if (json[valueStart] == '"')
            {
                // String value
                int end = json.IndexOf('"', valueStart + 1);
                if (end < 0) return "";
                return json.Substring(valueStart + 1, end - valueStart - 1);
            }
            else
            {
                // Numeric value
                int end = valueStart;
                while (end < json.Length && json[end] != ',' && json[end] != '\n'
                       && json[end] != '}' && json[end] != ' ')
                    end++;
                return json.Substring(valueStart, end - valueStart).Trim();
            }
        }

        private double ParseDouble(string value, double defaultValue)
        {
            double result;
            return double.TryParse(value, out result) ? result : defaultValue;
        }

        private int ParseInt(string value, int defaultValue)
        {
            int result;
            return int.TryParse(value, out result) ? result : defaultValue;
        }

        #endregion
    }
}
