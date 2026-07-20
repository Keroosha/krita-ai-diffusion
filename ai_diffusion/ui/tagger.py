from PyQt6.QtCore import QMetaObject
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..localization import translate as _
from ..model.model import DocumentModel
from ..model.properties import Bind, Binding, bind, bind_combo, bind_toggle
from ..model.root import root
from . import theme
from .theme import SignalBlocker
from .widget import ErrorBox, QueueButton, TextPromptWidget, WorkspaceSelectWidget


class TaggerWidget(QWidget):
    _model: DocumentModel
    _model_bindings: list[QMetaObject.Connection | Binding]

    def __init__(self):
        super().__init__()
        self._model = root.active_model
        self._model_bindings = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 0)

        self.workspace_select = WorkspaceSelectWidget(self)
        self.model_select = QComboBox(self)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.workspace_select)
        model_layout.addWidget(self.model_select)
        layout.addLayout(model_layout)

        self.threshold_input = self._create_threshold_input()
        self.character_threshold_input = self._create_threshold_input()
        self.replace_underscore_checkbox = QCheckBox(_("Replace underscores with spaces"), self)
        self.trailing_comma_checkbox = QCheckBox(_("Add trailing comma"), self)
        self.exclude_tags_input = QLineEdit(self)
        self.exclude_tags_input.setPlaceholderText(_("lowres, text, watermark"))

        form = QFormLayout()
        form.addRow(_("Threshold"), self.threshold_input)
        form.addRow(_("Character threshold"), self.character_threshold_input)
        form.addRow(self.replace_underscore_checkbox)
        form.addRow(self.trailing_comma_checkbox)
        form.addRow(_("Exclude tags"), self.exclude_tags_input)
        layout.addLayout(form)

        self.unavailable_warning = QLabel(
            _("WD14Tagger|pysssss is not installed or has an incompatible input contract."),
            self,
        )
        self.unavailable_warning.setWordWrap(True)
        self.unavailable_warning.setStyleSheet(f"color: {theme.yellow};")
        layout.addWidget(self.unavailable_warning)

        layout.addWidget(QLabel(_("Tags"), self))
        self.result_widget = TextPromptWidget(line_count=4, parent=self)
        self.result_widget.setPlaceholderText(_("Generated tags will appear here"))
        self.result_widget.text_changed.connect(self._update_result_actions)
        layout.addWidget(self.result_widget)

        self.copy_button = QPushButton(_("Copy"), self)
        self.copy_button.clicked.connect(self.copy_result)
        self.replace_prompt_button = QPushButton(_("Replace positive prompt"), self)
        self.replace_prompt_button.clicked.connect(self.replace_prompt)
        result_actions = QHBoxLayout()
        result_actions.addWidget(self.copy_button)
        result_actions.addWidget(self.replace_prompt_button)
        layout.addLayout(result_actions)

        self.tag_button = QPushButton(_("Tag"), self)
        self.tag_button.setMinimumHeight(32)
        self.tag_button.clicked.connect(self.tag_image)
        self.queue_button = QueueButton(supports_batch=False, parent=self)
        self.queue_button.setFixedHeight(self.tag_button.height() - 2)
        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.tag_button)
        actions_layout.addWidget(self.queue_button)
        layout.addLayout(actions_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.error_box = ErrorBox(self)
        layout.addWidget(self.error_box)
        layout.addStretch()

        self._parameter_widgets = (
            self.model_select,
            self.threshold_input,
            self.character_threshold_input,
            self.replace_underscore_checkbox,
            self.trailing_comma_checkbox,
            self.exclude_tags_input,
        )
        self._update_result_actions()
        self.model = self._model

    @staticmethod
    def _create_threshold_input():
        input = QDoubleSpinBox()
        input.setRange(0.0, 1.0)
        input.setSingleStep(0.05)
        input.setDecimals(2)
        return input

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: DocumentModel):
        if self._model == model and self._model_bindings:
            model.tagger.refresh_models()
            self.update_models()
            self._update_availability()
            return

        Binding.disconnect_all(self._model_bindings)
        self._model = model
        model.tagger.refresh_models()
        self.update_models()
        self._model_bindings = [
            bind(model, "workspace", self.workspace_select, "value", Bind.one_way),
            bind_combo(model.tagger, "model", self.model_select),
            bind(model.tagger, "threshold", self.threshold_input, "value"),
            bind(
                model.tagger,
                "character_threshold",
                self.character_threshold_input,
                "value",
            ),
            bind_toggle(
                model.tagger,
                "replace_underscore",
                self.replace_underscore_checkbox,
            ),
            bind_toggle(model.tagger, "trailing_comma", self.trailing_comma_checkbox),
            bind(model.tagger, "exclude_tags", self.exclude_tags_input, "text"),
            bind(model.tagger, "result", self.result_widget, "text"),
            bind(model.tagger, "can_tag", self.tag_button, "enabled", Bind.one_way),
            bind(model, "error", self.error_box, "error", Bind.one_way),
            model.tagger.models_changed.connect(self.update_models),
            model.tagger.is_available_changed.connect(self._update_availability),
            model.tagger.result_changed.connect(self._update_result_actions),
            model.progress_changed.connect(self.update_progress),
        ]
        self.queue_button.model = model
        self._update_availability()
        self._update_result_actions()
        self.update_progress()

    def update_models(self):
        with SignalBlocker(self.model_select):
            self.model_select.clear()
            for name in self.model.tagger.models:
                self.model_select.addItem(name, name)
            selected = self.model_select.findData(self.model.tagger.model)
            self.model_select.setCurrentIndex(selected)

    def _update_availability(self):
        available = self.model.tagger.is_available
        for widget in self._parameter_widgets:
            widget.setEnabled(available)
        self.unavailable_warning.setVisible(not available)

    def _update_result_actions(self):
        has_result = bool(self.result_widget.text.strip())
        self.copy_button.setEnabled(has_result)
        self.replace_prompt_button.setEnabled(has_result)

    def update_progress(self):
        self.progress_bar.setValue(int(self.model.progress * 100))

    def tag_image(self):
        self.model.tag_image()

    def copy_result(self):
        if clipboard := QGuiApplication.clipboard():
            clipboard.setText(self.result_widget.text)

    def replace_prompt(self):
        self.model.tagger.replace_prompt()
