"""Flat light theme shared with the SmartAnkle launcher (same palette, fonts and ttk styles)."""
import tkinter.font as tkfont
from tkinter import ttk

FONT_SIZE = 12
FONT_FAMILY = "Segoe UI"
COLORS = {
    "bg": "#eef2f7",
    "card": "#ffffff",
    "border": "#d5dce6",
    "text": "#1f2937",
    "muted": "#6b7280",
    "header": "#1e293b",
    "header_muted": "#94a3b8",
    "chip": "#273449",
    "unknown": "#64748b",
    "missing": "#f87171",
    "accent": "#2563eb",
    "accent_soft": "#dbe7fe",
    "accent_hover": "#1d4ed8",
    "warning": "#d97706",
    "success": "#16a34a",
    "success_hover": "#15803d",
    "danger": "#dc2626",
    "danger_hover": "#b91c1c",
    "disabled": "#e5e7eb",
    "log_bg": "#0f172a",
    "log_fg": "#cbd5e1",
    "running": "#facc15",
    "ready": "#4ade80",
}


def apply_theme(root):
    """ttk 'clam' recoloured like the SmartAnkle launcher."""
    for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(font_name).configure(family=FONT_FAMILY, size=FONT_SIZE)
    root.option_add("*TCombobox*Listbox.font", (FONT_FAMILY, FONT_SIZE))
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "white")
    root.configure(background=COLORS["bg"])

    style = ttk.Style(root)
    style.theme_use("clam")
    c = COLORS
    style.configure(".", background=c["bg"], foreground=c["text"], font=(FONT_FAMILY, FONT_SIZE),
                    bordercolor=c["border"], lightcolor=c["card"], darkcolor=c["card"],
                    focuscolor=c["accent"], troughcolor=c["bg"])

    style.configure("Header.TFrame", background=c["header"])
    style.configure("HeaderTitle.TLabel", background=c["header"], foreground="white",
                    font=(FONT_FAMILY, FONT_SIZE + 8, "bold"))
    style.configure("HeaderSub.TLabel", background=c["header"], foreground=c["header_muted"],
                    font=(FONT_FAMILY, FONT_SIZE - 2))
    style.configure("Status.TLabel", background=c["header"], foreground=c["header_muted"],
                    font=(FONT_FAMILY, FONT_SIZE - 1, "bold"))

    style.configure("Card.TFrame", background=c["card"], borderwidth=1, relief="solid")
    style.configure("Inner.TFrame", background=c["card"])
    style.configure("Card.TLabel", background=c["card"], foreground=c["muted"])
    style.configure("Value.TLabel", background=c["card"], foreground=c["text"],
                    font=(FONT_FAMILY, FONT_SIZE, "bold"))
    style.configure("CardTitle.TLabel", background=c["card"], foreground=c["text"],
                    font=(FONT_FAMILY, FONT_SIZE, "bold"))
    style.configure("Description.TLabel", background=c["bg"], foreground=c["muted"],
                    font=(FONT_FAMILY, FONT_SIZE - 2, "italic"))
    style.configure("Summary.TLabel", background=c["card"], foreground=c["muted"],
                    font=(FONT_FAMILY, FONT_SIZE - 3))
    style.configure("Footer.TLabel", background=c["bg"], foreground=c["muted"],
                    font=(FONT_FAMILY, FONT_SIZE - 4))
    style.configure("Section.TLabel", background=c["bg"], foreground=c["muted"],
                    font=(FONT_FAMILY, FONT_SIZE - 3, "bold"))

    style.configure("TNotebook", background=c["bg"], borderwidth=0, tabmargins=(24, 10, 24, 0))
    style.configure("TNotebook.Tab", background=c["bg"], foreground=c["muted"], padding=(22, 8),
                    borderwidth=0, font=(FONT_FAMILY, FONT_SIZE, "bold"))
    style.map("TNotebook.Tab", background=[("selected", c["card"])],
              foreground=[("selected", c["accent"])], lightcolor=[("selected", c["card"])])

    style.configure("Card.TCheckbutton", background=c["card"], foreground=c["text"], padding=(0, 1),
                    indicatorbackground=c["card"], indicatorforeground=c["card"],
                    indicatorsize=FONT_SIZE + 3, indicatormargin=(0, 0, 8, 0),
                    upperbordercolor=c["muted"], lowerbordercolor=c["muted"])
    style.map("Card.TCheckbutton",
              background=[("active", c["card"])],
              indicatorbackground=[("selected", c["accent"]), ("active", c["accent_soft"])],
              indicatorforeground=[("selected", "white")],
              upperbordercolor=[("selected", c["accent"])],
              lowerbordercolor=[("selected", c["accent"])])

    for widget_style in ("TEntry", "TSpinbox", "TCombobox"):
        style.configure(widget_style, fieldbackground=c["card"], foreground=c["text"], padding=4,
                        bordercolor=c["border"], lightcolor=c["card"], arrowcolor=c["muted"],
                        background=c["card"], selectbackground=c["accent"], selectforeground="white")
        style.map(widget_style, bordercolor=[("focus", c["accent"])], lightcolor=[("focus", c["accent"])])
    style.map("TCombobox",
              fieldbackground=[("readonly", c["card"])],
              selectbackground=[("readonly", c["card"])],
              selectforeground=[("readonly", c["text"])],
              bordercolor=[("focus", c["accent"])])

    def button_style(name, bg, fg, hover, border=None):
        style.configure(name, background=bg, foreground=fg, padding=(14, 6), relief="flat",
                        borderwidth=1, bordercolor=border or bg, lightcolor=bg, darkcolor=bg,
                        font=(FONT_FAMILY, FONT_SIZE, "bold"))
        style.map(name,
                  background=[("disabled", c["disabled"]), ("pressed", hover), ("active", hover)],
                  foreground=[("disabled", c["muted"])],
                  lightcolor=[("active", hover)], darkcolor=[("active", hover)],
                  bordercolor=[("disabled", c["disabled"])])

    button_style("Accent.TButton", c["success"], "white", c["success_hover"])
    button_style("Primary.TButton", c["accent"], "white", c["accent_hover"])
    button_style("Danger.TButton", c["danger"], "white", c["danger_hover"])
    button_style("Ghost.TButton", c["card"], c["accent"], c["accent_soft"], border=c["border"])

    style.configure("Treeview", background=c["card"], fieldbackground=c["card"], foreground=c["text"],
                    rowheight=FONT_SIZE * 2, bordercolor=c["border"])
    style.configure("Treeview.Heading", background=c["bg"], foreground=c["muted"], relief="flat",
                    font=(FONT_FAMILY, FONT_SIZE - 2, "bold"))
    style.map("Treeview", background=[("selected", c["accent_soft"])], foreground=[("selected", c["text"])])
