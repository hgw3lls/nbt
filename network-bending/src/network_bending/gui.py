"""Proof-of-concept GUI for exploring neural bends.

This lightweight Tkinter interface lets artists and researchers browse
bends by domain/category, read their philosophical intent, and run a
selected bend against a provided model. It stays intentionally simple to
keep the conceptual framing visible alongside the technical action.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

from .toolbox import apply_bend, list_bend_summaries


class MockModel:
    """Minimal placeholder model for demonstration purposes."""

    def __repr__(self) -> str:  # pragma: no cover - visualization only
        return "<MockModel — swap in your own network>"


class NetworkBendingGUI:
    """Small Tkinter GUI to surface bends and their media-theoretical aims."""

    def __init__(self, model_provider: Optional[Callable[[], Any]] = None) -> None:
        self.model_provider = model_provider or (lambda: MockModel())
        self.root = tk.Tk()
        self.root.title("Network Bending — PoC")
        self.root.geometry("780x520")

        self.domain_var = tk.StringVar(value="all")
        self.category_var = tk.StringVar(value="all")

        self.bend_summaries: List[Dict[str, str]] = []
        self._build_layout()
        self._refresh_bends()

    def run(self) -> None:  # pragma: no cover - UI loop
        """Start the Tkinter main loop."""

        self.root.mainloop()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        # Filters keep the philosophical framing legible: where do we act and to what end?
        ttk.Label(controls, text="Domain").pack(side=tk.LEFT)
        self.domain_menu = ttk.Combobox(
            controls,
            textvariable=self.domain_var,
            values=self._domain_options(),
            state="readonly",
            width=18,
        )
        self.domain_menu.pack(side=tk.LEFT, padx=5)
        self.domain_menu.bind("<<ComboboxSelected>>", lambda _: self._refresh_bends())

        ttk.Label(controls, text="Category").pack(side=tk.LEFT)
        self.category_menu = ttk.Combobox(
            controls,
            textvariable=self.category_var,
            values=self._category_options(),
            state="readonly",
            width=18,
        )
        self.category_menu.pack(side=tk.LEFT, padx=5)
        self.category_menu.bind("<<ComboboxSelected>>", lambda _: self._refresh_bends())

        refresh_button = ttk.Button(controls, text="Refresh", command=self._refresh_bends)
        refresh_button.pack(side=tk.RIGHT)

        content = ttk.Frame(self.root, padding=10)
        content.pack(fill=tk.BOTH, expand=True)

        self.bend_list = tk.Listbox(content, height=15, exportselection=False)
        self.bend_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.bend_list.bind("<<ListboxSelect>>", self._show_details)

        self.detail_text = tk.Text(content, wrap=tk.WORD)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.detail_text.configure(state="disabled")

        action_bar = ttk.Frame(self.root, padding=10)
        action_bar.pack(fill=tk.X)

        self.apply_button = ttk.Button(action_bar, text="Apply bend", command=self._apply_selected)
        self.apply_button.pack(side=tk.RIGHT)

    def _domain_options(self) -> List[str]:
        domains = sorted({b["domain"] for b in list_bend_summaries()})
        return ["all"] + domains

    def _category_options(self) -> List[str]:
        categories = sorted({b["category"] for b in list_bend_summaries()})
        return ["all"] + categories

    def _refresh_bends(self) -> None:
        domain_filter = None if self.domain_var.get() == "all" else self.domain_var.get()
        category_filter = None if self.category_var.get() == "all" else self.category_var.get()
        self.bend_summaries = list_bend_summaries(domain=domain_filter, category=category_filter)

        self.bend_list.delete(0, tk.END)
        for bend in self.bend_summaries:
            label = f"{bend['name']}  [{bend['domain']} · {bend['category']}]"
            self.bend_list.insert(tk.END, label)

        if self.bend_summaries:
            self.bend_list.select_set(0)
            self._show_details()
        else:
            self._set_detail_text("No bends match the current filters.")

    def _set_detail_text(self, text: str) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text)
        self.detail_text.configure(state="disabled")

    def _show_details(self, event: Optional[tk.Event] = None) -> None:  # pragma: no cover - UI callback
        if not self.bend_summaries:
            return
        selection = self.bend_list.curselection()
        if not selection:
            return
        idx = selection[0]
        bend = self.bend_summaries[idx]
        detail = (
            f"Name: {bend['name']}\n"
            f"Domain: {bend['domain']}\n"
            f"Category: {bend['category']}\n\n"
            f"Description:\n{bend['description']}\n"
        )
        self._set_detail_text(detail)

    def _apply_selected(self) -> None:  # pragma: no cover - UI callback
        if not self.bend_summaries:
            messagebox.showinfo("No bend", "Choose a bend to apply.")
            return
        selection = self.bend_list.curselection()
        if not selection:
            messagebox.showinfo("No bend", "Select a bend first.")
            return
        bend = self.bend_summaries[selection[0]]
        model = self.model_provider()
        try:
            result = apply_bend(bend["name"], model)
            messagebox.showinfo(
                "Bend applied",
                f"Applied '{bend['name']}'.\n\n"
                f"Result model: {result.model}\n"
                f"Metadata keys: {list(result.metadata)}",
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Error", f"Failed to apply bend: {exc}")


def launch_gui(model_provider: Optional[Callable[[], Any]] = None) -> None:
    """Launch the proof-of-concept GUI.

    Parameters
    ----------
    model_provider: Optional[Callable[[], Any]]
        Function that returns the model to bend when the user clicks
        "Apply bend". Defaults to a :class:`MockModel` placeholder.
    """

    gui = NetworkBendingGUI(model_provider=model_provider)
    gui.run()


if __name__ == "__main__":  # pragma: no cover
    launch_gui()
