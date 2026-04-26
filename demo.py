import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time
import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


VIDEO_SYNTACTIC_L1 = r"c:\Users\agost\Downloads\L1--success=False--task=position_the_bowl_on_the_top_of_the_cabinet.mp4"
VIDEO_SYNTACTIC_L2 = r"c:\Users\agost\Downloads\L2--episode=58--success=False--task=the_top_of_the_drawer_needs_to_have_the_bowl_on_it.mp4"
VIDEO_SYNTACTIC_L3 = r"c:\Users\agost\Downloads\L3--success=False--task=put_the_object_in_front_of_the_wine_bottle_on_the_.mp4"
VIDEO_TASK_L1      = r"c:\Users\agost\Downloads\TaskL1--episode=169--success=False--task=put_the_cream_cheese_on_the_plate.mp4"
VIDEO_TASK_L2      = r"c:\Users\agost\Downloads\TaskL2--episode=132--success=False--task=put_the_cream_cheese_on_the_bowl_and_put_the_bowl_on_the_plate.mp4"
VIDEO_TRAINING_1   = r"c:\Users\agost\Downloads\Train--episode=87--success=True--task=put_the_bowl_on_the_top_of_the_cabinet.mp4"
VIDEO_TRAINING_2   = r"c:\Users\agost\Downloads\TaskTrain--episode=450--success=True--task=put_the_bowl_on_the_plate.mp4"
VIDEO_FINETUNED_L3 = r"c:\Users\agost\Downloads\Finetune--episode=66--success=True--task=put_the_object_in_front_of_the_wine_bottle_on_the_.mp4" 

TEST_CONFIG = {
    "Syntactic Generalisation - L1  |  Verb Substitution":                  (VIDEO_SYNTACTIC_L1, False),
    "Syntactic Generalisation - L2  |  Syntactic Restructuring":             (VIDEO_SYNTACTIC_L2, False),
    "Syntactic Generalisation - L3  |  Compositional Spatial Ref.":          (VIDEO_SYNTACTIC_L3, False),
    "Syntactic Generalisation - L3  |  Finetuned Model":      (VIDEO_FINETUNED_L3, True),
    "Task Generalisation - L1  |  Cross Object Skill Transfer":              (VIDEO_TASK_L1,      False),
    "Task Generalisation - L2  |  Novel Task Composition":                   (VIDEO_TASK_L2,      False),
    "Training Task":                                                          (VIDEO_TRAINING_1,   True),
    "Training Task":                                                          (VIDEO_TRAINING_2,   True),
}


BG_ROOT       = "#f5f4f0"
BG_PANEL      = "#ffffff"
BG_TERM       = "#0d1117"
BG_INPUT      = "#f0eeea"
BORDER        = "#dbd8d0"
ACCENT        = "#c8732e"
ACCENT_HVR    = "#a85c20"
GREEN_UI      = "#2e7d52"
YELLOW_UI     = "#c49a10"
TEXT_DARK     = "#3b1a0a"
TEXT_DIM      = "#888899"
WHITE         = "#ffffff"
COLOR_SUCCESS = "#27ae60"
COLOR_FAIL    = "#e74c3c"


T_USER   = "#ff9f43"
T_SYSTEM = "#48dbfb"
T_VLA    = "#a29bfe"
T_SIM    = "#55efc4"
T_OK     = "#00ff88"
T_INFO   = "#ffd32a"
T_ERROR  = "#ff4757"
T_DIM    = "#3d5068"
T_WHITE  = "#e8edf2"


LOG_STEPS = [
    ("system", 0.70, "[SYSTEM]  Command received and validated"),
    ("system", 0.60, "[SYSTEM]  Sending language instruction to VLA model..."),
    ("system", 0.55, "[SYSTEM]  Sending camera images of the scene..."),
    ("system", 0.55, "[SYSTEM]  Sending robot joint states..."),
    ("vla",    0.45, "[VLA]  ✓  Language instruction received"),
    ("vla",    0.45, "[VLA]  ✓  Camera images received"),
    ("vla",    0.45, "[VLA]  ✓  Robot joint states received"),
    ("vla",    0.65, "[VLA]     Starting language preprocessing (tokenisation)..."),
    ("vla",    0.55, "[VLA]     Text encoding — language encoder (LLM backbone)..."),
    ("vla",    0.55, "[VLA]     Vision tokens — scene observation encoder (ViT)..."),
    ("vla",    0.55, "[VLA]     State tokens  — robot proprioception encoder..."),
    ("vla",    0.65, "[VLA]     Multimodal fusion (Transformer backbone)..."),
    ("vla",    0.50, "[VLA]  ✓  Multimodal representation assembled"),
    ("vla",    0.70, "[VLA]     Action decoder processing (Diffusion / Flow)..."),
    ("vla",    0.50, "[VLA]  ✓  Action chunk generated"),
    ("vla",    0.40, "[VLA]  ✓  Action sequence transmitted to robot controller"),
    ("sim",    0.35, "[SIM]     Executing trajectory in LIBERO environment..."),
]


class VLADemo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LIBERO - Benchmarking Knowledge Transfer for Lifelong Robot Learning")
        try:
            _raw = Image.open(
                r"C:\Users\agost\Documents\GitHub\Evaluating-Generalization-of-State-of-the-Art-Multi-Task-Language-Conditioned-Imitation-Learning-Systems\LIBERO\images\libero_logo.png"
            ).convert("RGBA")
            _raw.thumbnail((64, 64), Image.LANCZOS)
            _canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            _offset = ((64 - _raw.width) // 2, (64 - _raw.height) // 2)
            _canvas.paste(_raw, _offset)
            _icon_photo = ImageTk.PhotoImage(_canvas)
            self.iconphoto(True, _icon_photo)
            self._icon_ref = _icon_photo
        except Exception as e:
            print(f"[WARN] Icona non caricata: {e}")

        self.configure(bg=BG_ROOT)
        self.resizable(True, True)
        self.minsize(1180, 700)

        self._active_video   = None
        self._active_success = False

        self._playing       = False
        self._first_frame   = None
        self._result_shown  = False
        self._video_w       = 640
        self._video_h       = 480

        self._build_fonts()
        self._build_ui()

    def _build_fonts(self):
        self.f_title   = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.f_section = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.f_label   = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.f_btn     = tkfont.Font(family="Helvetica", size=13, weight="bold")
        self.f_combo   = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.f_mono_md = tkfont.Font(family="Courier",   size=13, weight="bold")
        self.f_mono_sm = tkfont.Font(family="Courier",   size=12, weight="bold")
        self.f_mono_xs = tkfont.Font(family="Courier",   size=11, weight="bold")
        self.f_result  = tkfont.Font(family="Helvetica", size=24, weight="bold")

    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=BG_PANEL, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="LIBERO Simulator Interface",
                 font=self.f_title, bg=BG_PANEL, fg=ACCENT).pack(
                     side="left", padx=22, pady=14)
        tk.Label(hdr, text="LIBERO Goal Suite",
                 font=self.f_label, bg=BG_PANEL, fg=TEXT_DIM).pack(side="left")
        self._hdr_status = tk.Label(hdr, text="READY",
                                    font=self.f_label, bg=BG_PANEL, fg=GREEN_UI)
        self._hdr_status.pack(side="right", padx=22)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # PanedWindow
        self._paned = tk.PanedWindow(self, orient="horizontal",
                                     bg=BORDER, sashwidth=5,
                                     sashrelief="flat", opaqueresize=True)
        self._paned.pack(fill="both", expand=True)
        left  = self._build_left()
        right = self._build_right()
        self._paned.add(left,  minsize=380, width=580)
        self._paned.add(right, minsize=400)

    def _build_left(self):
        frame = tk.Frame(self._paned, bg=BG_ROOT)
        sec   = tk.Frame(frame, bg=BG_ROOT)
        sec.pack(fill="x", padx=18, pady=(18, 0))

        tk.Label(sec, text="TASK COMMAND", font=self.f_section,
                 bg=BG_ROOT, fg=ACCENT).pack(anchor="w")
        tk.Frame(sec, bg=ACCENT, height=2).pack(fill="x", pady=(3, 10))

        tk.Label(sec, text="Generalisation Level / Test Case",
                 font=self.f_label, bg=BG_ROOT, fg=TEXT_DIM).pack(anchor="w")

        self._combo_var = tk.StringVar()
        combo_style = ttk.Style()
        combo_style.theme_use("clam")
        combo_style.configure("VLA.TCombobox",
            fieldbackground=BG_INPUT, background=BG_INPUT,
            foreground=TEXT_DARK, arrowcolor=ACCENT,
            selectbackground=BG_INPUT, selectforeground=TEXT_DARK,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        combo_style.map("VLA.TCombobox",
            fieldbackground=[("readonly", BG_INPUT), ("focus", BG_INPUT)],
            foreground=[("readonly", TEXT_DARK), ("focus", TEXT_DARK)],
            selectbackground=[("readonly", BG_INPUT), ("focus", BG_INPUT)],
            selectforeground=[("readonly", TEXT_DARK), ("focus", TEXT_DARK)])

        self._combo = ttk.Combobox(
            sec,
            textvariable=self._combo_var,
            values=list(TEST_CONFIG.keys()),
            state="readonly",
            style="VLA.TCombobox",
            font=self.f_combo,
            width=58
        )
        self._combo.pack(fill="x", pady=(2, 12))
        self._combo.bind("<<ComboboxSelected>>", self._on_combo_select)

        tk.Label(sec, text="Language Instruction",
                 font=self.f_label, bg=BG_ROOT, fg=TEXT_DIM).pack(anchor="w")

        cmd_border = tk.Frame(sec, bg=BORDER)
        cmd_border.pack(fill="x", pady=(2, 0))
        cmd_inner = tk.Frame(cmd_border, bg=BG_INPUT)
        cmd_inner.pack(fill="x", padx=1, pady=1)
        row = tk.Frame(cmd_inner, bg=BG_INPUT)
        row.pack(fill="x", padx=10, pady=6)
        tk.Label(row, text="❯ ", font=self.f_mono_md,
                 bg=BG_INPUT, fg=ACCENT).pack(side="left")
        self._cmd_var = tk.StringVar()
        self._cmd_entry = tk.Entry(
            row, textvariable=self._cmd_var,
            font=self.f_mono_md,
            bg=BG_INPUT, fg="#3b1a0a",
            insertbackground=ACCENT,
            relief="flat", bd=0)
        self._cmd_entry.pack(side="left", fill="x", expand=True)
        self._cmd_entry.bind("<Return>", lambda e: self._on_send())

        tk.Frame(sec, bg=BORDER, height=1).pack(fill="x", pady=(0, 8))

        self._send_btn = tk.Button(
            sec, text="▶  SEND TO VLA MODEL",
            font=self.f_btn,
            bg=ACCENT, fg=WHITE,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", bd=0,
            padx=14, pady=8,
            cursor="hand2",
            command=self._on_send)
        self._send_btn.pack(fill="x")

        sec_t = tk.Frame(frame, bg=BG_ROOT)
        sec_t.pack(fill="both", expand=True, padx=18, pady=(20, 16))

        hdr_row = tk.Frame(sec_t, bg=BG_ROOT)
        hdr_row.pack(fill="x")
        tk.Label(hdr_row, text="TERMINAL LOG", font=self.f_section,
                 bg=BG_ROOT, fg=ACCENT).pack(side="left")
        tk.Button(hdr_row, text="CLEAR",
                  font=self.f_label, bg=BG_ROOT, fg=TEXT_DIM,
                  activebackground=BORDER, activeforeground=TEXT_DARK,
                  relief="flat", bd=0, cursor="hand2",
                  command=self._clear_log).pack(side="right")
        tk.Frame(sec_t, bg=ACCENT, height=2).pack(fill="x", pady=(3, 6))

        term_outer = tk.Frame(sec_t, bg=BORDER)
        term_outer.pack(fill="both", expand=True)
        term_inner = tk.Frame(term_outer, bg=BG_TERM)
        term_inner.pack(fill="both", expand=True, padx=1, pady=1)

        tbar = tk.Frame(term_inner, bg="#090d12", height=28)
        tbar.pack(fill="x")
        tbar.pack_propagate(False)
        for dot_col in ("#e74c3c", "#f39c12", "#27ae60"):
            tk.Label(tbar, text="●", font=self.f_mono_xs,
                     bg="#090d12", fg=dot_col).pack(
                         side="left",
                         padx=(8 if dot_col == "#e74c3c" else 2, 0))
        tk.Label(tbar, text="Log Terminal",
                 font=self.f_mono_xs, bg="#090d12", fg=T_DIM).pack(
                     side="left", padx=12)

        self._term = tk.Text(
            term_inner,
            bg=BG_TERM, fg=T_WHITE,
            font=self.f_mono_sm,
            relief="flat", bd=0,
            state="disabled", wrap="word",
            padx=12, pady=10,
            cursor="arrow",
            insertbackground=ACCENT,
            selectbackground="#1c3a5e")
        sb = tk.Scrollbar(term_inner, command=self._term.yview,
                          bg=BG_TERM, troughcolor="#090d12",
                          activebackground="#1c3a5e", bd=0, relief="flat")
        self._term.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._term.pack(side="left", fill="both", expand=True)

        # tag colori
        self._term.tag_configure("user",   foreground=T_USER)
        self._term.tag_configure("system", foreground=T_SYSTEM)
        self._term.tag_configure("vla",    foreground=T_VLA)
        self._term.tag_configure("sim",    foreground=T_SIM)
        self._term.tag_configure("ok",     foreground=T_OK)
        self._term.tag_configure("info",   foreground=T_INFO)
        self._term.tag_configure("error",  foreground=T_ERROR)
        self._term.tag_configure("dim",    foreground=T_DIM)
        self._term.tag_configure("white",  foreground=T_WHITE)

        self._term_append("dim", "  ┌───────── LIBERO Simulator Interface ─────────────")
        self._term_append("dim", "  │  Ready. Select a test case and enter a command.")
        self._term_append("dim", "  └──────────────────────────────────────────────────\n")

        return frame

    def _build_right(self):
        frame = tk.Frame(self._paned, bg=BG_ROOT)

        info_row = tk.Frame(frame, bg=BG_ROOT)
        info_row.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(info_row, text="FRONT CAMERA VIEW", font=self.f_section,
                 bg=BG_ROOT, fg=ACCENT).pack(side="left")
        self._vid_status = tk.Label(info_row, text="● Recording",
                                    font=self.f_label, bg=BG_ROOT, fg=COLOR_FAIL)
        self._vid_status.pack(side="right")
        tk.Frame(frame, bg=ACCENT, height=2).pack(fill="x", padx=20, pady=(4, 0))

        self._vid_container = tk.Frame(frame, bg=BG_ROOT)
        self._vid_container.pack(fill="both", expand=True, padx=20, pady=14)

        self._vid_label = tk.Label(self._vid_container, bg=BG_ROOT,
                                   relief="flat", bd=0)
        self._vid_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        return frame

    def _term_append(self, tag, text):
        self._term.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._term.insert("end", f"[{ts}]  ", "dim")
        self._term.insert("end", text + "\n", tag)
        self._term.configure(state="disabled")
        self._term.see("end")

    def _clear_log(self):
        self._term.configure(state="normal")
        self._term.delete("1.0", "end")
        self._term.configure(state="disabled")

    def _on_combo_select(self, _event=None):
        key = self._combo_var.get()
        video_path, success = TEST_CONFIG[key]
        self._active_video   = video_path
        self._active_success = success
        self._playing = False
        self._hide_result_overlay()
        self._reset_video_panel()
        self._load_first_frame(video_path)

    def _reset_video_panel(self):
        self._vid_label.configure(image="", bg=BG_ROOT)
        self._vid_label.image = None

    def _load_first_frame(self, path=None):
        if path is None:
            return
        scale = 2.0
        if not (HAS_CV2 and HAS_PIL):
            return
        if not os.path.isfile(path):
            self._term_append("error", f"[ERROR]  File not found: {path}")
            return
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._first_frame = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._video_w = int(self._first_frame.size[0] * scale)
            self._video_h = int(self._first_frame.size[1] * scale)
            self._vid_container.configure(
                width=self._video_w, height=self._video_h)
            self._vid_container.pack_propagate(False)
            self._display_pil(self._first_frame)

    def _display_pil(self, pil_img):
        iw, ih = pil_img.size
        if hasattr(self, "_video_w"):
            if iw != self._video_w or ih != self._video_h:
                pil_img = pil_img.resize(
                    (self._video_w, self._video_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self._vid_label.configure(image=photo)
        self._vid_label.image = photo

    def _show_result_overlay(self):
        if hasattr(self, "_result_canvas") and \
                self._result_canvas.winfo_exists():
            self._result_canvas.destroy()

        self._vid_container.update_idletasks()

        cw = self._video_w
        ch = self._video_h

        container_w = self._vid_container.winfo_width()
        container_h = self._vid_container.winfo_height()
        x = max(0, (container_w - cw) // 2)
        y = max(0, (container_h - ch) // 2)

        color = COLOR_SUCCESS if self._active_success else COLOR_FAIL
        label = ("✓  Correct Execution"
                if self._active_success else "✗  Wrong Execution")

        c = tk.Canvas(self._vid_container,
                    width=cw, height=ch,
                    bg=BG_ROOT,
                    highlightthickness=0, bd=0)
        c.place(x=x, y=y)
        self._vid_label.lower(c)
        self._result_canvas = c

        if self._last_frame:
            resized = self._last_frame.resize((cw, ch), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            c.create_image(0, 0, anchor="nw", image=photo)
            c._bg_photo = photo

        bw = 10
        c.create_rectangle(bw // 2, bw // 2,
                            cw - bw // 2, ch - bw // 2,
                            outline=color, width=bw, fill="")

        pad_x, pad_y = 28, 10
        txt_x, txt_y = cw // 2, 38
        tmp = c.create_text(txt_x, txt_y, text=label,
                            font=self.f_result, fill=WHITE, anchor="center")
        bb = c.bbox(tmp)
        c.delete(tmp)
        if bb:
            x1, y1, x2, y2 = bb
            c.create_rectangle(x1 - pad_x, y1 - pad_y,
                                x2 + pad_x, y2 + pad_y,
                                fill=color, outline="", width=0)
        c.create_text(txt_x, txt_y, text=label,
                    font=self.f_result, fill=WHITE, anchor="center")
        self._result_shown = True

    def _hide_result_overlay(self):
        if hasattr(self, "_result_canvas") and \
                self._result_canvas.winfo_exists():
            self._result_canvas.destroy()
        self._result_shown = False

    def _hide_result_overlay(self):
        if hasattr(self, "_result_canvas") and \
                self._result_canvas.winfo_exists():
            self._result_canvas.destroy()
        self._result_shown = False

    def _on_send(self):
        cmd = self._cmd_var.get().strip()
        if not cmd:
            self._term_append("error",
                "[ERROR]  Empty command — enter a task instruction.")
            return
        if not self._active_video:
            self._term_append("error",
                "[ERROR]  Select a test case from the dropdown first.")
            return
        if self._playing:
            self._term_append("error",
                "[ERROR]  Simulation already running, please wait.")
            return
        self._send_btn.configure(state="disabled",
                                 bg="#d0c0b0", fg=TEXT_DIM)
        self._hdr_status.configure(text="PROCESSING", fg=YELLOW_UI)
        self._hide_result_overlay()
        threading.Thread(target=self._run_pipeline,
                         args=(cmd,), daemon=True).start()
        
    def _reset_to_first_frame(self):
        """Nasconde l'overlay risultato e ripristina il primo frame del video."""
        self._hide_result_overlay()
        if self._first_frame:
            self._display_pil(self._first_frame)

    def _run_pipeline(self, cmd):
        tokens = cmd.lower().split()
        self.after(0, self._term_append, "user",  f"[USER]   ❯ {cmd}")
        self.after(0, self._term_append, "dim",   "─" * 54)

        delay_ms = 200
        for (tag, secs, text) in LOG_STEPS:
            delay_ms += int(secs * 1000)
            if "Starting language preprocessing" in text:
                self.after(delay_ms - 50, self._term_append, "info",
                        f"[VLA]     Extracted tokens: {tokens}")
            self.after(delay_ms, self._term_append, tag, text)

        delay_ms += 350
        self.after(delay_ms, self._start_video)
        delay_ms += 600
        self.after(delay_ms, self._reenable_send)

    def _reenable_send(self):
        self._cmd_var.set("") 
        self._send_btn.configure(state="normal", bg=ACCENT, fg=WHITE)

    def _start_video(self):
        if not (HAS_CV2 and HAS_PIL):
            self._term_append("error",
                "[ERROR]  opencv-python / Pillow not installed.")
            return
        if not self._active_video or \
                not os.path.isfile(self._active_video):
            self._term_append("error",
                f"[ERROR]  Video not found: {self._active_video}")
            return
        self._playing      = True
        self._result_shown = False
        self._hdr_status.configure(text="RUNNING",  fg=GREEN_UI)
        threading.Thread(target=self._play_video, daemon=True).start()

    def _play_video(self):
        cap   = cv2.VideoCapture(self._active_video)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps
        last_pil = None

        while self._playing:
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_pil = pil_img
            self.after(0, self._display_pil, pil_img)
            time.sleep(delay)

        cap.release()
        self._playing = False

        if last_pil is not None:
            self._last_frame = last_pil

        success = self._active_success
        if success:
            result_tag  = "vla"
            result_text = "[VLA]  ✓  Episode complete — Correct Execution"
            status_col  = COLOR_SUCCESS
        else:
            result_tag  = "error"
            result_text = "[VLA]  ✗  Episode complete — Wrong Execution"
            status_col  = COLOR_FAIL

        self.after(0, self._hdr_status.configure,
                   {"text": "WAITING FOR INPUT", "fg": YELLOW_UI})
        self.after(0, self._term_append, result_tag, result_text)
        self.after(0, self._term_append, "dim", "─" * 54)
        self.after(0, self._show_result_overlay)

        if self._first_frame:
            self.after(400, self._display_pil, self._first_frame)


if __name__ == "__main__":
    missing = []
    if not HAS_CV2:  missing.append("opencv-python")
    if not HAS_PIL:  missing.append("Pillow")
    if missing:
        print(f"[ERROR] Missing: pip install {' '.join(missing)}")
        raise SystemExit(1)
    VLADemo().mainloop()