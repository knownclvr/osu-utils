using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.WebSockets;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

sealed class ManiaNote
{
    public int Lane;
    public double Start, End;
    public bool Hold;
}

sealed class KeyTransition
{
    public int Lane;
    public double Time, End;
    public bool Down, Hold;
}

sealed class ManiaChart
{
    public string Title;
    public int Notes, Conflicts;
    public List<ManiaNote> Holds = new List<ManiaNote>();
    public List<KeyTransition> Events = new List<KeyTransition>();
    public static ManiaChart Parse(string text)
    {
        string section = "", title = "", version = "";
        int mode = -1, keyCount = 0;
        var notes = new List<ManiaNote>();
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line.StartsWith("//"))
                continue;
            if (line.StartsWith("["))
            {
                section = line;
                continue;
            }
            int colon = line.IndexOf(':');
            if (section == "[General]" && line.StartsWith("Mode:"))
                mode = int.Parse(line.Substring(colon + 1).Trim());
            if (section == "[Difficulty]" && line.StartsWith("CircleSize:"))
                keyCount = (int)double.Parse(line.Substring(colon + 1).Trim(), CultureInfo.InvariantCulture);
            if (section == "[Metadata]" && line.StartsWith("Title:"))
                title = line.Substring(colon + 1).Trim();
            if (section == "[Metadata]" && line.StartsWith("Version:"))
                version = line.Substring(colon + 1).Trim();
            if (section != "[HitObjects]")
                continue;
            string[] fields = line.Split(',');
            if (fields.Length < 5)
                continue;
            int x = int.Parse(fields[0]), type = int.Parse(fields[3]);
            double time = double.Parse(fields[2], CultureInfo.InvariantCulture);
            bool hold = (type & 128) != 0;
            double end = hold && fields.Length > 5
                             ? double.Parse(fields[5].Split(':')[0], CultureInfo.InvariantCulture)
                             : time;
            if (x < 0 || x > 512 || end < time)
                throw new InvalidDataException("잘못된 노트 좌표 또는 롱노트 시간");
            notes.Add(new ManiaNote { Lane = Math.Min(3, x * 4 / 512), Start = time, End = end,
                                      Hold = hold && end > time });
        }
        if (mode != 3 || keyCount != 4)
            throw new InvalidDataException("이 도구는 osu!mania 4K 맵만 지원합니다.");
        var beatmap = new ManiaChart { Title = title + " [" + version + "]", Notes = notes.Count };
        for (int lane = 0; lane < 4; lane++)
        {
            var columnNotes = notes.Where(note => note.Lane == lane)
                                  .OrderBy(note => note.Start)
                                  .ThenByDescending(note => note.End)
                                  .ToList();
            double heldUntil = double.NegativeInfinity;
            for (int i = 0; i < columnNotes.Count; i++)
            {
                ManiaNote note = columnNotes[i];
                if (note.Start < heldUntil)
                {
                    beatmap.Conflicts++;
                    continue;
                }
                double gap = i + 1 < columnNotes.Count ? columnNotes[i + 1].Start - note.Start : 100;
                double end = note.Hold ? note.End : note.Start + Math.Max(0.5, Math.Min(24, gap * 0.65));
                heldUntil = end;
                if (note.Hold)
                    beatmap.Holds.Add(note);
                beatmap.Events.Add(new KeyTransition { Lane = lane, Time = note.Start, End = end, Down = true,
                                                       Hold = note.Hold });
                beatmap.Events.Add(
                    new KeyTransition { Lane = lane, Time = end, End = end, Down = false, Hold = note.Hold });
            }
        }
        beatmap.Events =
            beatmap.Events.OrderBy(e => e.Time).ThenBy(e => e.Down ? 1 : 0).ThenBy(e => e.Lane).ToList();
        return beatmap;
    }
}

static class Win32Input
{
    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT
    {
        public uint type;
        public UNION U;
    }
    [StructLayout(LayoutKind.Explicit)]
    public struct UNION
    {
        [FieldOffset(0)]
        public MOUSE mi;
        [FieldOffset(0)]
        public KEY ki;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSE
    {
        public int x, y;
        public uint data, flags, time;
        public UIntPtr extra;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct KEY
    {
        public ushort vk, scan;
        public uint flags, time;
        public UIntPtr extra;
    }
    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint SendInput(uint count, INPUT[] inputs, int size);
    [DllImport("user32.dll")]
    public static extern uint MapVirtualKey(uint code, uint mapType);
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr handle, out uint id);
    [DllImport("user32.dll")]
    public static extern bool RegisterHotKey(IntPtr handle, int id, uint modifiers, uint key);
    [DllImport("user32.dll")]
    public static extern bool UnregisterHotKey(IntPtr handle, int id);
    [DllImport("winmm.dll")]
    public static extern uint timeBeginPeriod(uint ms);
    [DllImport("winmm.dll")]
    public static extern uint timeEndPeriod(uint ms);
    public static bool IsGameFocused()
    {
        uint id;
        GetWindowThreadProcessId(GetForegroundWindow(), out id);
        if (id == lastForegroundPid)
            return lastForegroundWasOsu;
        lastForegroundPid = id;
        lastForegroundWasOsu = false;
        try
        {
            using (var p = Process.GetProcessById((int)id)) return lastForegroundWasOsu =
                p.ProcessName.Equals("osu!", StringComparison.OrdinalIgnoreCase) ||
                p.ProcessName.Equals("osu", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    static uint lastForegroundPid;
    static bool lastForegroundWasOsu;
}

sealed class AutoplayerHost : Form
{
    readonly object stateLock = new object();
    readonly CancellationTokenSource shutdown = new CancellationTokenSource();
    readonly Stopwatch uptime = Stopwatch.StartNew();
    readonly JavaScriptSerializer gameJson = new JavaScriptSerializer();
    const int ExitHotkeyId = 19;
    readonly bool[] keyDown = new bool[4];
    readonly Queue<KeyTransition>[] keyQueues =
        Enumerable.Range(0, 4).Select(i => new Queue<KeyTransition>()).ToArray();
    readonly double[] lastPressAtMs = new double[4],
                      lastReleaseAtMs = Enumerable.Repeat(double.NegativeInfinity, 4).ToArray();
    readonly ushort[] scanCodes = new ushort[4];
    readonly ushort[] keys = { (ushort)Keys.C, (ushort)Keys.V, (ushort)Keys.M, (ushort)Keys.Oemcomma };
    readonly string tosuUrl;
    Process analysisProcess;
    TcpListener overlayListener;
    volatile string overlayJson = "{}";
    string mapTier = "계산 중", sectionTier = "계산 중";
    bool inputActive, isClosing, tosuConnected, isPlaying, wasPlaying, needsResync, modsSupported = true;
    double songTimeMs, lastStateAtMs, lastSongMoveAtMs, rate = 1, lastInputTimeMs;
    double preciseTimeMs, lastPreciseAtMs = double.NegativeInfinity,
                          lastPreciseMoveAtMs = double.NegativeInfinity;
    bool hasPreciseClock;
    bool preciseSocketConnected, waitingForClock;
    double clockWaitUntilMs;
    const double LateGraceMs = 80, MinimumPressMs = 8, MinimumReleaseMs = 2;
    string beatmapPath = "", statusReason = "tosu 연결 중...", connectionLabel = "연결 중";
    ManiaChart beatmap;
    int nextEventIndex, submittedNotes, lateNotes, lastInputError;
    readonly Thread inputThread;
    readonly System.Windows.Forms.Timer overlayTimer;

    public AutoplayerHost(string url)
    {
        tosuUrl = url;
        Text = "오수핵";
        ShowInTaskbar = false;
        gameJson.MaxJsonLength = 8 * 1024 * 1024;
        var hudSerializer = new JavaScriptSerializer();
        overlayTimer = new System.Windows.Forms.Timer { Interval = 100 };
        overlayTimer.Tick += delegate
        {
            object snapshot;
            lock (stateLock) snapshot = CreateOverlayState();
            overlayJson = hudSerializer.Serialize(snapshot);
        };
        inputThread = new Thread(RunInputLoop) { IsBackground = true, Name = "osu input",
                                                 Priority = ThreadPriority.AboveNormal };
        CreateHandle();
        for (int i = 0; i < 4; i++)
            scanCodes[i] = (ushort)Win32Input.MapVirtualKey(keys[i], 0);
        overlayTimer.Start();
        Win32Input.timeBeginPeriod(1);
        inputThread.Start();
        Task.Run(() => ReadGameStateAsync());
        Task.Run(() => ReadPreciseClockAsync());
        Task.Run(() => RunAnalysisAsync());
        Task.Run(() => RunOverlayServerAsync());
    }

    protected override void SetVisibleCore(bool value)
    {
        base.SetVisibleCore(false);
    }

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        if (!Win32Input.RegisterHotKey(Handle, ExitHotkeyId, 0x4002, (uint)Keys.OemCloseBrackets))
            throw new InvalidOperationException("Ctrl + ] 종료 단축키를 등록할 수 없습니다.");
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == 0x312 && (int)m.WParam == ExitHotkeyId)
            Close();
        base.WndProc(ref m);
    }

    static object ReadField(object obj, params string[] path)
    {
        foreach (string key in path)
        {
            var fields = obj as Dictionary<string, object>;
            if (fields == null || !fields.TryGetValue(key, out obj))
                return null;
        }
        return obj;
    }

    static string AsString(object value)
    {
        return Convert.ToString(value, CultureInfo.InvariantCulture);
    }

    static double AsNumber(object value, double fallback = 0)
    {
        double number;
        return double.TryParse(AsString(value), NumberStyles.Float, CultureInfo.InvariantCulture, out number)
                   ? number
                   : fallback;
    }

    static bool SupportsMods(int bits)
    {
        return (bits & ~(1 | 8 | 16 | 32 | 64 | 256 | 512 | 16384)) == 0;
    }

    object CreateOverlayState()
    {
        return new { updatedAt = (long)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
                                     .TotalMilliseconds,
                     active = !isClosing && tosuConnected && beatmap != null,
                     map = beatmap == null ? "" : beatmap.Title,
                     whole = mapTier,
                     segment = sectionTier,
                     running = inputActive && isPlaying && !isClosing };
    }

    static byte[] CreateHttpResponse(string body)
    {
        byte[] payload = Encoding.UTF8.GetBytes(body);
        byte[] header = Encoding.ASCII.GetBytes(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nCache-Control: " +
            "no-store\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Private-Network: " +
            "true\r\nAccess-Control-Allow-Methods: GET, OPTIONS\r\nContent-Length: " + payload.Length +
            "\r\nConnection: close\r\n\r\n");
        return header.Concat(payload).ToArray();
    }

    async Task RunOverlayServerAsync(int port = 24051)
    {
        var listener = new TcpListener(IPAddress.Loopback, port);
        lock (stateLock)
        {
            if (isClosing)
                return;
            overlayListener = listener;
            listener.Start();
        }
        try
        {
            while (!shutdown.IsCancellationRequested)
            {
                var client = await listener.AcceptTcpClientAsync();
                await ReplyOverlayRequestAsync(client);
            }
        }
        catch (SocketException)
        {
        }
        catch (ObjectDisposedException)
        {
        }
        finally
        {
            listener.Stop();
        }
    }

    async Task ReplyOverlayRequestAsync(TcpClient client)
    {
        using (client) using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(shutdown.Token))
        {
            timeout.CancelAfter(1500);
            try
            {
                var stream = client.GetStream();
                byte[] buffer = new byte[4096];
                int used = 0;
                while (used < buffer.Length)
                {
                    int count = await stream.ReadAsync(buffer, used, buffer.Length - used, timeout.Token);
                    if (count == 0)
                        return;
                    used += count;
                    if (Encoding.ASCII.GetString(buffer, 0, used).Contains("\r\n\r\n"))
                        break;
                }
                string request = Encoding.ASCII.GetString(buffer, 0, used);
                if (!request.StartsWith("GET /status.json ") && !request.StartsWith("OPTIONS /status.json "))
                    return;
                byte[] response = CreateHttpResponse(overlayJson);
                await stream.WriteAsync(response, 0, response.Length, timeout.Token);
            }
            catch (IOException)
            {
            }
            catch (OperationCanceledException)
            {
            }
            catch (ObjectDisposedException)
            {
            }
        }
    }

    async Task RunAnalysisAsync()
    {
        var serializer = new JavaScriptSerializer();
        while (!shutdown.IsCancellationRequested)
        {
            Process process = null;
            try
            {
                string root = AppDomain.CurrentDomain.BaseDirectory;
                string node = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                                           "nodejs", "node.exe");
                if (!File.Exists(node))
                    node = "node.exe";
                var info = new ProcessStartInfo(node, "\"" + Path.Combine(root, "dan-bridge.mjs") +
                                                          "\"") { UseShellExecute = false,
                                                                    CreateNoWindow = true,
                                                                    WorkingDirectory = root,
                                                                    RedirectStandardInput = true,
                                                                    RedirectStandardOutput = true,
                                                                    RedirectStandardError = true,
                                                                    StandardOutputEncoding = Encoding.UTF8,
                                                                    StandardErrorEncoding = Encoding.UTF8 };
                process = new Process { StartInfo = info };
                process.ErrorDataReceived += delegate {};
                process.Start();
                process.BeginErrorReadLine();
                try
                {
                    process.PriorityClass = ProcessPriorityClass.BelowNormal;
                }
                catch (System.ComponentModel.Win32Exception)
                {
                }
                lock (stateLock)
                {
                    if (isClosing)
                    {
                        process.Kill();
                        return;
                    }
                    analysisProcess = process;
                }
                long requestId = 0;
                string lastKey = "";
                while (!shutdown.IsCancellationRequested && !process.HasExited)
                {
                    string file;
                    double speed, time;
                    bool ready;
                    lock (stateLock)
                    {
                        file = beatmapPath;
                        speed = rate;
                        time = songTimeMs;
                        ready = tosuConnected && beatmap != null && modsSupported;
                    }
                    if (ready)
                    {
                        string key = file + "|" + speed.ToString("R", CultureInfo.InvariantCulture) + "|" +
                                     Math.Floor(Math.Max(0, time) / (500 * speed));
                        if (key != lastKey)
                        {
                            byte[] request = Encoding.UTF8.GetBytes(serializer.Serialize(new {
                                id = ++requestId, path = file, rate = speed, timeMs = time
                            }) + "\n");
                            await process.StandardInput.BaseStream.WriteAsync(request, 0, request.Length);
                            await process.StandardInput.BaseStream.FlushAsync();
                            Task<string> read = process.StandardOutput.ReadLineAsync();
                            if (await Task.WhenAny(read, Task.Delay(15000, shutdown.Token)) != read)
                                throw new IOException("단 계산 시간 초과");
                            string line = await read;
                            if (line == null)
                                throw new IOException("단 계산 엔진 종료");
                            var result = serializer.DeserializeObject(line);
                            if (AsNumber(ReadField(result, "id")) != requestId)
                                throw new IOException("단 계산 응답 불일치");
                            lock (stateLock)
                            {
                                if (!isClosing && beatmapPath == file && rate == speed)
                                {
                                    mapTier = AsString(ReadField(result, "whole"));
                                    sectionTier = AsString(ReadField(result, "segment"));
                                    lastKey = key;
                                }
                            }
                        }
                    }
                    await Task.Delay(500, shutdown.Token);
                }
            }
            catch (Exception)
            {
                lock (stateLock)
                {
                    mapTier = sectionTier = "계산 불가";
                }
            }
            finally
            {
                lock (stateLock) if (analysisProcess == process) analysisProcess = null;
                if (process != null)
                {
                    try
                    {
                        if (!process.HasExited)
                            process.Kill();
                    }
                    catch (InvalidOperationException)
                    {
                    }
                    process.Dispose();
                }
            }
            if (!shutdown.IsCancellationRequested)
                try
                {
                    await Task.Delay(2000, shutdown.Token);
                }
                catch (OperationCanceledException)
                {
                }
        }
    }

    async Task ReadGameStateAsync()
    {
        while (!shutdown.IsCancellationRequested)
        {
            using (var socket = new ClientWebSocket())
            try
            {
                await socket.ConnectAsync(new Uri(tosuUrl), shutdown.Token);
                lock (stateLock)
                {
                    tosuConnected = true;
                    connectionLabel = "tosu 연결됨";
                    statusReason = "자동 대기";
                }
                var filter = Encoding.UTF8.GetBytes(
                    "applyFilters:[{\"field\":\"state\",\"keys\":[\"number\"]},{\"field\":\"folders\"," +
                    "\"keys\":[\"songs\",\"beatmap\"]},{\"field\":\"files\",\"keys\":[\"beatmap\"]},{" +
                    "\"field\":\"beatmap\",\"keys\":[{\"field\":\"time\",\"keys\":[\"live\"]}]},{\"field\":" +
                    "\"play\",\"keys\":[\"failed\",{\"field\":\"mods\",\"keys\":[\"number\",\"rate\"]}]}]");
                await socket.SendAsync(new ArraySegment<byte>(filter), WebSocketMessageType.Text, true,
                                       shutdown.Token);
                byte[] buffer = new byte[65536];
                while (socket.State == WebSocketState.Open && !shutdown.IsCancellationRequested)
                {
                    using (var data = new MemoryStream())
                    {
                        WebSocketReceiveResult r;
                        do
                        {
                            r = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), shutdown.Token);
                            if (r.MessageType == WebSocketMessageType.Close)
                                throw new IOException("tosu 연결 종료");
                            data.Write(buffer, 0, r.Count);
                            if (data.Length > gameJson.MaxJsonLength)
                                throw new IOException("너무 큰 메시지");
                        } while (!r.EndOfMessage);
                        UpdateGameState(gameJson.DeserializeObject(Encoding.UTF8.GetString(data.ToArray())));
                    }
                }
            }
            catch (Exception ex)
            {
                if (!shutdown.IsCancellationRequested)
                    lock (stateLock)
                    {
                        tosuConnected = false;
                        isPlaying = false;
                        StopInput("tosu 연결 끊김");
                        connectionLabel = ex is OperationCanceledException ? "연결 취소" : "재연결 대기";
                    }
            }
            if (!shutdown.IsCancellationRequested)
                try
                {
                    await Task.Delay(1500, shutdown.Token);
                }
                catch (OperationCanceledException)
                {
                }
        }
    }

    async Task ReadPreciseClockAsync()
    {
        var serializer = new JavaScriptSerializer();
        var uri = new UriBuilder(tosuUrl) { Path = "/websocket/v2/precise", Query = "" }.Uri;
        while (!shutdown.IsCancellationRequested)
        {
            using (var socket = new ClientWebSocket())
            try
            {
                await socket.ConnectAsync(uri, shutdown.Token);
                lock (stateLock) preciseSocketConnected = true;
                byte[] filter = Encoding.UTF8.GetBytes("applyFilters:[\"currentTime\"]");
                await socket.SendAsync(new ArraySegment<byte>(filter), WebSocketMessageType.Text, true,
                                       shutdown.Token);
                byte[] buffer = new byte[4096];
                while (socket.State == WebSocketState.Open && !shutdown.IsCancellationRequested)
                {
                    using (var data = new MemoryStream())
                    {
                        WebSocketReceiveResult r;
                        do
                        {
                            r = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), shutdown.Token);
                            if (r.MessageType == WebSocketMessageType.Close)
                                throw new IOException("Precise clock disconnected");
                            data.Write(buffer, 0, r.Count);
                            if (data.Length > 65536)
                                throw new IOException("Precise clock message too large");
                        } while (!r.EndOfMessage);
                        object d = serializer.DeserializeObject(Encoding.UTF8.GetString(data.ToArray()));
                        object value = ReadField(d, "currentTime");
                        double time;
                        if (value != null && double.TryParse(AsString(value), NumberStyles.Float,
                                                             CultureInfo.InvariantCulture, out time))
                            lock (stateLock) UpdatePreciseClock(time, uptime.Elapsed.TotalMilliseconds);
                    }
                }
            }
            catch (Exception)
            {
                lock (stateLock)
                {
                    hasPreciseClock = false;
                    preciseSocketConnected = false;
                    waitingForClock = false;
                }
            }
            if (!shutdown.IsCancellationRequested)
                try
                {
                    await Task.Delay(1000, shutdown.Token);
                }
                catch (OperationCanceledException)
                {
                }
        }
    }

    void UpdatePreciseClock(double time, double now)
    {
        if (!tosuConnected || !isPlaying || beatmap == null || double.IsNaN(time) || double.IsInfinity(time))
            return;
        double coarse = songTimeMs + Math.Min(150, Math.Max(0, now - lastSongMoveAtMs)) * rate;
        // 맵 전환 직후 들어온 이전 시간값은 버린다.
        if (Math.Abs(time - coarse) > 250)
        {
            hasPreciseClock = false;
            return;
        }
        if (!hasPreciseClock || time != preciseTimeMs)
            lastPreciseMoveAtMs = now;
        preciseTimeMs = time;
        lastPreciseAtMs = now;
        hasPreciseClock = true;
        waitingForClock = false;
    }

    void UpdateGameState(object data)
    {
        if (ReadField(data, "error") != null)
            return;
        string file = AsString(ReadField(data, "files", "beatmap"));
        string songs = AsString(ReadField(data, "folders", "songs")),
               folder = AsString(ReadField(data, "folders", "beatmap"));
        string filename = "";
        if (songs.Length > 0 && file.Length > 0)
            filename = Path.GetFullPath(Path.Combine(songs, folder, file));
        bool changed;
        lock (stateLock) changed = filename != beatmapPath;
        ManiaChart loaded = null;
        string error = null;
        if (changed && filename.Length > 0)
            try
            {
                loaded = ManiaChart.Parse(File.ReadAllText(filename));
            }
            catch (Exception ex)
            {
                error = ex.Message;
            }
        lock (stateLock)
        {
            if (isClosing)
                return;
            if (changed)
            {
                StopInput("맵 변경");
                hasPreciseClock = false;
                beatmapPath = filename;
                beatmap = loaded;
                mapTier = sectionTier = "계산 중";
                if (error != null)
                    statusReason = error;
            }
            if (ReadField(data, "beatmap", "time", "live") == null)
                return;
            double now = uptime.Elapsed.TotalMilliseconds;
            double next = AsNumber(ReadField(data, "beatmap", "time", "live"));
            bool nextPlaying = AsNumber(ReadField(data, "state", "number")) == 2;
            bool enteringPlay = !isPlaying && nextPlaying;
            bool seeking = isPlaying && (next < songTimeMs - 100 ||
                                         Math.Abs(next - (songTimeMs + (now - lastStateAtMs) * rate)) > 500);
            if (next != songTimeMs)
                lastSongMoveAtMs = now;
            if (seeking)
            {
                StopInput("재생시간 이동");
                hasPreciseClock = false;
            }
            // 실패 플래그가 남아 있어도 현재 플레이 상태를 우선한다.
            if (!nextPlaying)
            {
                if (inputActive && wasPlaying)
                    StopInput("플레이 종료 또는 일시정지");
                hasPreciseClock = false;
            }
            songTimeMs = next;
            lastStateAtMs = now;
            isPlaying = nextPlaying;
            double explicitRate = AsNumber(ReadField(data, "play", "mods", "rate"));
            int bits = (int)AsNumber(ReadField(data, "play", "mods", "number"));
            modsSupported = SupportsMods(bits);
            if (inputActive && !modsSupported)
                StopInput("지원하지 않는 패턴 변경 모드");
            double nextRate = explicitRate > 0           ? explicitRate
                              : (bits & (64 | 512)) != 0 ? 1.5
                              : (bits & 256) != 0        ? 0.75
                                                         : 1;
            bool rateChanged = nextRate != rate;
            if (rateChanged)
            {
                StopInput("속도 변경");
                hasPreciseClock = false;
                mapTier = sectionTier = "계산 중";
            }
            rate = nextRate;
            if (nextPlaying && (enteringPlay || seeking || changed || rateChanged))
            {
                // 시작할 때는 첫 정밀 시간값을 받은 뒤 입력한다.
                StopInput("시작 시간 동기화");
                needsResync = true;
                hasPreciseClock = false;
                waitingForClock = preciseSocketConnected;
                clockWaitUntilMs = now + 40;
            }
            if (!nextPlaying)
                waitingForClock = false;
            if (inputActive && isPlaying)
                wasPlaying = true;
        }
    }

    bool SendKey(int lane, bool down)
    {
        if (keyDown[lane] == down)
            return true;

        {
            ushort scan = scanCodes[lane];
            var input = new Win32Input.INPUT { type = 1, U = new Win32Input.UNION { ki = new Win32Input.KEY {
                                                  scan = scan, flags = (uint)(0x0008 | (down ? 0 : 0x0002))
                                              } } };
            if (Win32Input.SendInput(1, new[] { input }, Marshal.SizeOf(typeof(Win32Input.INPUT))) != 1)
            {
                lastInputError = Marshal.GetLastWin32Error();
                if (lastInputError == 0)
                    lastInputError = -1;
                return false;
            }
        }
        keyDown[lane] = down;
        return true;
    }

    void ReleaseAllKeys()
    {
        for (int i = 0; i < 4; i++)
        {
            keyQueues[i].Clear();
            if (keyDown[i])
                SendKey(i, false);
            lastReleaseAtMs[i] = double.NegativeInfinity;
        }
    }

    void StopInput(string why)
    {
        inputActive = false;
        wasPlaying = false;
        ReleaseAllKeys();
        statusReason = why;
    }

    void PlayUntil(double estimate, double now)
    {
        if (needsResync)
        {
            // 중간에 재개하면 아직 끝나지 않은 롱노트부터 잡는다.
            double seekFrom = estimate - LateGraceMs * rate;
            int lo = 0, hi = beatmap.Events.Count;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (beatmap.Events[mid].Time < seekFrom)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            nextEventIndex = lo;
            foreach (var h in beatmap.Holds)
                if (h.Start < seekFrom && h.End > estimate)
                    keyQueues[h.Lane].Enqueue(new KeyTransition { Lane = h.Lane, Time = estimate, End = h.End,
                                                                  Down = true, Hold = true });
            lastInputTimeMs = estimate;
            needsResync = false;
        }
        estimate = Math.Max(estimate, lastInputTimeMs);
        lastInputTimeMs = estimate;
        while (inputActive && nextEventIndex < beatmap.Events.Count &&
               beatmap.Events[nextEventIndex].Time <= estimate)
        {
            var e = beatmap.Events[nextEventIndex++];
            if (e.Down && estimate - e.Time > LateGraceMs * rate && (!e.Hold || e.End <= estimate))
            {
                lateNotes++;
                continue;
            }
            keyQueues[e.Lane].Enqueue(e);
        }
        SendQueuedKeys(now, estimate);
        if (inputActive)
            statusReason = "입력 중";
    }

    void SendQueuedKeys(double now, double estimate)
    {
        var actions = new List<KeyTransition>();
        for (int lane = 0; lane < 4; lane++)
        {
            var q = keyQueues[lane];
            while (q.Count > 0 && q.Peek().Down == keyDown[lane])
                q.Dequeue();
            if (q.Count == 0)
                continue;
            var e = q.Peek();
            if (e.Down && estimate - e.Time > LateGraceMs * rate && (!e.Hold || e.End <= estimate))
            {
                q.Dequeue();
                lateNotes++;
                continue;
            }
            if (e.Down ? now - lastReleaseAtMs[lane] < MinimumReleaseMs
                       : now - lastPressAtMs[lane] < MinimumPressMs)
                continue;
            actions.Add(e);
        }
        int inserted = actions.Count;
        if (actions.Count > 0)
        {
            var inputs =
                actions
                    .Select(e => new Win32Input.INPUT {
                        type = 1,
                        U =
                            new Win32Input.UNION {
                                ki = new Win32Input.KEY { scan = scanCodes[e.Lane],
                                                          flags = (uint)(0x0008 | (e.Down ? 0 : 0x0002)) }
                            }
                    })
                    .ToArray();
            inserted = (int)Win32Input.SendInput((uint)inputs.Length, inputs,
                                                 Marshal.SizeOf(typeof(Win32Input.INPUT)));
        }
        for (int i = 0; i < inserted; i++)
        {
            var e = actions[i];
            keyQueues[e.Lane].Dequeue();
            keyDown[e.Lane] = e.Down;

            if (e.Down)
            {
                lastPressAtMs[e.Lane] = now;
                submittedNotes++;
            }
            else
                lastReleaseAtMs[e.Lane] = now;
        }
        if (inserted != actions.Count)
        {
            lastInputError = Marshal.GetLastWin32Error();
            StopInput("키 입력 실패");
        }
    }

    void RunInputLoop()
    {
        while (!shutdown.IsCancellationRequested)
        {
            lock (stateLock)
            {
                UpdateInput(Win32Input.IsGameFocused(), uptime.Elapsed.TotalMilliseconds);
            }
            Thread.Sleep(1);
        }
    }

    void UpdateInput(bool focused, double now)
    {
        bool usePrecise = hasPreciseClock && now - lastPreciseAtMs <= 60;
        bool ready = !isClosing && focused && tosuConnected && isPlaying && beatmap != null &&
                     modsSupported && now - lastStateAtMs <= 300 && now - lastSongMoveAtMs <= 180 &&
                     (!usePrecise || now - lastPreciseMoveAtMs <= 80);
        if (!ready)
        {
            StopInput("대기");
            needsResync = true;
            return;
        }
        if (waitingForClock)
        {
            if (now < clockWaitUntilMs)
            {
                ReleaseAllKeys();
                inputActive = false;
                needsResync = true;
                return;
            }
            waitingForClock = false; // 정밀 시간이 안 오면 일반 시간으로 복귀한다.
        }
        if (!inputActive)
        {
            inputActive = true;
            needsResync = true;
            submittedNotes = lateNotes = 0;
            lastInputError = 0;
        }
        // 같은 시간값이 반복돼도 보간 기준점은 유지한다.
        double estimate = usePrecise
                              ? preciseTimeMs + Math.Min(30, Math.Max(0, now - lastPreciseMoveAtMs)) * rate
                              : songTimeMs + Math.Min(150, Math.Max(0, now - lastSongMoveAtMs)) * rate;
        PlayUntil(estimate, now);
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        lock (stateLock)
        {
            isClosing = true;
            StopInput("종료");
        }
        shutdown.Cancel();
        if (inputThread.IsAlive)
            inputThread.Join(1500);
        overlayTimer.Stop();
        if (overlayListener != null)
            overlayListener.Stop();
        lock (stateLock) if (analysisProcess != null)
        try
        {
            if (!analysisProcess.HasExited)
                analysisProcess.Kill();
        }
        catch (InvalidOperationException)
        {
        }
        {
            Win32Input.UnregisterHotKey(Handle, ExitHotkeyId);
            Win32Input.timeEndPeriod(1);
        }
        base.OnFormClosing(e);
    }

    [STAThread]
    static int Main(string[] args)
    {
        bool first;
        using (var mutex = new Mutex(true, "Local\\TosuOfflineTest4K", out first))
        {
            if (!first)
                return 0;
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            string tosuUrl = "ws://127.0.0.1:24050/websocket/v2";
            string cfg = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "server.txt");
            if (File.Exists(cfg))
                tosuUrl = File.ReadAllText(cfg).Trim();
            Uri uri;
            if (!Uri.TryCreate(tosuUrl, UriKind.Absolute, out uri) || uri.Scheme != "ws" || !uri.IsLoopback)
            {
                File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "startup-error.txt"),
                                  "server.txt에는 로컬 WebSocket 주소만 사용할 수 있습니다.");
                return 1;
            }
            try
            {
                Application.Run(new AutoplayerHost(tosuUrl));
                return 0;
            }
            catch (Exception ex)
            {
                File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "startup-error.txt"),
                                  ex.Message);
                return 1;
            }
        }
    }
}
