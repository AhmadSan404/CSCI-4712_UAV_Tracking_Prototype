"""The tkinter menu where the video, models and options are chosen before a run starts."""

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.camera_scanner import list_cameras
from src.detection.model_catalog import list_model_choices
from src.menu.menu_variables import NO_AERIAL_LABEL, NO_MAIN_LABEL, OTHER_VIDEO_LABEL, MenuVariables
from src.menu.tracker_settings import (
    CONFIDENCE_STEP,
    DEVICE_CHOICES,
    IMAGE_SIZE_STEP_PX,
    MAX_CONFIDENCE,
    MAX_IMAGE_SIZE_PX,
    MIN_CONFIDENCE,
    MIN_IMAGE_SIZE_PX,
    NO_MODEL,
    SOURCE_STREAM,
    SOURCE_VIDEO,
    TrackerSettings,
)
from src.project_paths import VIDEO_EXTENSIONS

PADDING_PX = 6
VIDEO_ENTRY_WIDTH_CHARS = 70
MODEL_BOX_WIDTH_CHARS = 80
SCALE_LENGTH_PX = 320


@dataclass
class MenuResult:
    """What the menu returns when Start is pressed: the settings and the screen size in pixels."""

    settings: TrackerSettings
    screen_width: int
    screen_height: int


def show_error_dialog(title: str, message: str) -> None:
    """Shows an error message box (with a temporary hidden window, since no menu is open)."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message, parent=root)
    root.destroy()


class MenuWindow:
    """The menu window. `show` blocks until the user presses Start or Quit."""

    def __init__(self, settings: TrackerSettings) -> None:
        self._result: MenuResult | None = None
        self._row_count = 0

        self._root = tk.Tk()
        self._root.title("Drone Car Tracker")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._quit)
        self._frame = ttk.Frame(self._root, padding=12)
        self._frame.grid()

        self._model_choices = list_model_choices()
        self._variables = MenuVariables(self._model_choices)
        self._build_layout()
        self._variables.apply(settings)
        self._scan_cameras()
        self._update_field_states()

    def show(self) -> MenuResult | None:
        """Runs the menu; returns the chosen settings, or None if the user quit."""
        self._root.mainloop()
        return self._result

    def _build_layout(self) -> None:
        """Adds one row per option, then the buttons."""
        variables = self._variables
        main_labels = [choice.label for choice in variables.main_choices]
        aerial_labels = [choice.label for choice in variables.aerial_choices]

        holder = self._add_row("Source")
        for text, value in (("Video file", SOURCE_VIDEO), ("Live stream (camera)", SOURCE_STREAM)):
            ttk.Radiobutton(
                holder, text=text, value=value, variable=variables.source_type, command=self._update_field_states
            ).pack(side="left", padx=(0, PADDING_PX * 2))

        holder = self._add_row("Video")
        self._video_box = video_box = ttk.Combobox(
            holder, textvariable=variables.video_choice, values=variables.video_names + [OTHER_VIDEO_LABEL],
            state="readonly", width=MODEL_BOX_WIDTH_CHARS,
        )
        video_box.pack(anchor="w", pady=2)
        video_box.bind("<<ComboboxSelected>>", lambda _event: self._update_field_states())
        path_row = ttk.Frame(holder)
        path_row.pack(anchor="w", pady=2)
        self._video_entry = ttk.Entry(path_row, textvariable=variables.video_path, width=VIDEO_ENTRY_WIDTH_CHARS)
        self._video_entry.pack(side="left")
        self._browse_button = ttk.Button(path_row, text="Browse...", command=self._browse_video)
        self._browse_button.pack(side="left", padx=PADDING_PX)

        holder = self._add_row("Camera")
        self._camera_box = ttk.Combobox(
            holder, textvariable=variables.camera_name, state="readonly", width=VIDEO_ENTRY_WIDTH_CHARS
        )
        self._camera_box.pack(side="left")
        self._scan_button = ttk.Button(holder, text="Scan cameras", command=self._scan_cameras)
        self._scan_button.pack(side="left", padx=PADDING_PX)

        main_box = self._add_model_dropdown("Main model", variables.main_model_label, [NO_MAIN_LABEL] + main_labels)
        main_box.bind("<<ComboboxSelected>>", lambda _event: self._update_field_states())
        self._add_image_size("Main model size (px)", variables.main_image_size_px)
        self._add_model_dropdown("Aerial model", variables.aerial_model_label, [NO_AERIAL_LABEL] + aerial_labels)
        self._add_image_size("Aerial model size (px)", variables.aerial_image_size_px)

        holder = self._add_row("Confidence floor")
        tk.Scale(
            holder, variable=variables.min_confidence, from_=MIN_CONFIDENCE, to=MAX_CONFIDENCE,
            resolution=CONFIDENCE_STEP, orient="horizontal", length=SCALE_LENGTH_PX,
        ).pack(anchor="w")

        holder = self._add_row("Classes")
        for name, flag in variables.class_flags.items():
            ttk.Checkbutton(holder, text=name, variable=flag).pack(side="left", padx=PADDING_PX)

        self._track_people_check = ttk.Checkbutton(self._add_row(""), text="Track people", variable=variables.track_people)
        self._track_people_check.pack(anchor="w")
        ttk.Checkbutton(self._add_row(""), text="Show trails", variable=variables.show_trails).pack(anchor="w")
        speed_text = "Start at max speed (uncapped)"
        self._max_speed_check = ttk.Checkbutton(
            self._add_row(""), text=speed_text, variable=variables.should_start_at_max_speed
        )
        self._max_speed_check.pack(anchor="w")

        device_box = ttk.Combobox(
            self._add_row("Device"), textvariable=variables.device_label, state="readonly", width=10,
            values=[choice.upper() for choice in DEVICE_CHOICES],
        )
        device_box.pack(anchor="w")
        self._build_buttons()

    def _add_row(self, label_text: str) -> ttk.Frame:
        """Adds a label in the left column and returns an empty frame in the right column for the controls."""
        row = self._row_count
        self._row_count += 1
        ttk.Label(self._frame, text=label_text).grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=PADDING_PX // 2)
        holder = ttk.Frame(self._frame)
        holder.grid(row=row, column=1, sticky="w")
        return holder

    def _add_model_dropdown(self, label_text: str, variable: tk.StringVar, labels: list[str]) -> ttk.Combobox:
        """A read-only dropdown of model labels."""
        box = ttk.Combobox(self._add_row(label_text), textvariable=variable, values=labels, state="readonly")
        box.config(width=MODEL_BOX_WIDTH_CHARS)
        box.pack(anchor="w", pady=2)
        return box

    def _add_image_size(self, label_text: str, variable: tk.IntVar) -> None:
        """A spinbox for a model's input size."""
        spinbox = ttk.Spinbox(
            self._add_row(label_text), from_=MIN_IMAGE_SIZE_PX, to=MAX_IMAGE_SIZE_PX, increment=IMAGE_SIZE_STEP_PX,
            textvariable=variable, width=8, state="readonly",
        )
        spinbox.pack(anchor="w", pady=2)

    def _build_buttons(self) -> None:
        """Start, Reset to defaults and Quit."""
        buttons = ttk.Frame(self._frame)
        for text, command in (("Start", self._start), ("Reset to defaults", self._reset), ("Quit", self._quit)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=PADDING_PX)
        buttons.grid(row=self._row_count, column=0, columnspan=2, pady=(12, 0))

    def _update_field_states(self) -> None:
        """Grays out the controls that do not apply: the video ones while streaming, the camera ones otherwise, and
        'Track people' while the main model is off (the aerial model has no person class).

        The path box and Browse button are only enabled while 'Other...' is selected in the video dropdown.
        """
        variables = self._variables
        is_stream = variables.source_type.get() == SOURCE_STREAM
        is_other = variables.video_choice.get() == OTHER_VIDEO_LABEL
        self._video_box.config(state="disabled" if is_stream else "readonly")
        self._video_entry.config(state="normal" if is_other and not is_stream else "disabled")
        self._browse_button.config(state="normal" if is_other and not is_stream else "disabled")
        self._camera_box.config(state="readonly" if is_stream else "disabled")
        self._scan_button.config(state="normal" if is_stream else "disabled")
        self._max_speed_check.config(state="disabled" if is_stream else "normal")  # a live source cannot speed up
        has_main_model = variables.main_model_label.get() != NO_MAIN_LABEL
        self._track_people_check.config(state="normal" if has_main_model else "disabled")

    def _scan_cameras(self) -> None:
        """Lists the connected cameras in the camera dropdown."""
        names = [camera.name for camera in list_cameras()]
        self._variables.set_camera_names(names)
        self._camera_box.config(values=names)

    def _browse_video(self) -> None:
        """Lets the user pick a video file from anywhere on the PC."""
        pattern = " ".join(f"*{extension}" for extension in sorted(VIDEO_EXTENSIONS))
        typed_path = self._variables.video_path.get().strip()
        current_folder = Path(typed_path).parent if typed_path else Path.home()
        chosen = filedialog.askopenfilename(
            initialdir=current_folder if current_folder.is_dir() else Path.home(),
            filetypes=[("Video files", pattern), ("All files", "*.*")],
        )
        if chosen:
            self._variables.video_path.set(str(Path(chosen)))

    def _start(self) -> None:
        """Validates the choices and closes the menu with the result."""
        settings = self._variables.collect()
        if settings.source_type == SOURCE_STREAM and not settings.camera_name:
            message = "No camera found. Start the OBS Virtual Camera (or connect a camera) and press Scan cameras."
            messagebox.showerror("No camera", message, parent=self._root)
            return
        if settings.source_type == SOURCE_VIDEO and not Path(settings.video_path).is_file():
            messagebox.showerror("No video", "Please choose a video file that exists.", parent=self._root)
            return
        if settings.main_model_name == NO_MODEL and settings.aerial_model_name == NO_MODEL:
            messagebox.showerror("No model", "Please choose a main model, an aerial model, or both.", parent=self._root)
            return
        if not settings.selected_class_names:
            messagebox.showerror("No classes", "Please tick at least one class to detect.", parent=self._root)
            return
        self._result = MenuResult(settings, self._root.winfo_screenwidth(), self._root.winfo_screenheight())
        self._root.destroy()

    def _reset(self) -> None:
        """Restores every option to its default."""
        self._variables.apply(TrackerSettings())
        self._update_field_states()

    def _quit(self) -> None:
        """Closes the menu without starting."""
        self._result = None
        self._root.destroy()


def show_menu(settings: TrackerSettings) -> MenuResult | None:
    """Opens the menu prefilled with `settings`; returns the choices, or None if the user quit."""
    return MenuWindow(settings).show()
