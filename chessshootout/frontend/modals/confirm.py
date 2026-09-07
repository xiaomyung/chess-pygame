from collections.abc import Callable
from typing import Any, cast

import pygame as pg

from chessshootout.frontend.modals.base import BaseModal, MODAL_MAX_WIDTH, MODAL_RAIL
from chessshootout.frontend.visual.colors import Colors
from chessshootout.frontend.visual.draw import cut_rect_surface
from chessshootout.frontend.visual.emoji import blit_emoji
from chessshootout.frontend.visual.fonts import fonts_for_width, get_display_font, get_font
from chessshootout.frontend.visual.widgets import draw_button_row, fit_text_to_rect, wrap_words


TITLE_SUB_GAP = 8
TITLE_TILE_CUT = 10
SUB_MAX_LINES = 3


class ConfirmModal(BaseModal):
    """
    The app's yes/no question box, asked before anything the player cannot
    undo: resigning, offering a draw, quitting, moving the saved games folder.
    It is a global modal, so it outranks whatever the active screen owns for
    Esc, for clicks and in the draw order, and it wears the shared modal shell
    """

    def __init__(self, window: pg.Surface) -> None:
        """
        Build the single question box the whole app reuses. It starts with no
        question set, which is also how it reports itself as hidden

        :param window: the app window surface this modal draws onto
        """
        super().__init__(window)
        self.title: str | None = None
        self.sub = ""
        self.danger = False
        self.emoji: str | None = None
        self.on_yes: Callable[[], None] | None = None
        self.on_no: Callable[[], None] | None = None
        self.on_extra: Callable[[], None] | None = None
        self.yes_label = "Confirm"
        self.no_label = "Cancel"
        self.extra_label = "Cancel"
        self.button_rects: dict[str, pg.Rect] = {}
        self._panel = pg.Rect(0, 0, 0, 0)
        self._font_cache: dict[str, Any] = {}

    def show(self, title: str, on_yes: Callable[[], None],
             on_no: Callable[[], None] | None = None, yes_label: str = "Confirm",
             no_label: str = "Cancel", on_extra: Callable[[], None] | None = None,
             extra_label: str = "Cancel", sub: str = "", danger: bool = False,
             emoji: str | None = None) -> None:
        """
        Ask the player a question and remember what each answer should do.
        Setting a title is what makes the box visible, so one call both fills
        it in and puts it on screen

        :param title: the question itself, drawn large and upper-cased
        :param on_yes: run when the confirming button is clicked
        :param on_no: run when the cancelling button is clicked, or None to
            just close the box; an empty no_label leaves that button out, and
            with it any way for this to run
        :param yes_label: text on the confirming button
        :param no_label: text on the cancelling button, or empty to leave that
            button out and state the one answer there is
        :param on_extra: run by a third button, or None to leave that button
            out entirely
        :param extra_label: text on that third button
        :param sub: smaller explanation under the title, wrapped over a few
            lines, empty for none
        :param danger: True to tint the rail and icon tile red for an answer
            that cannot be taken back
        :param emoji: emoji drawn in a tile above the title, or None for no icon
        """
        self.title = title
        self.sub = sub
        self.danger = danger
        self.emoji = emoji
        self.on_yes = on_yes
        self.on_no = on_no
        self.on_extra = on_extra
        self.yes_label = yes_label
        self.no_label = no_label
        self.extra_label = extra_label

    def hide(self) -> None:
        """
        Close the box and forget the question along with all three callbacks,
        so an answer can never fire twice or land in a game that has moved on
        """
        self.title = None
        self.emoji = None
        self.on_yes = None
        self.on_no = None
        self.on_extra = None
        self.button_rects = {}

    def is_visible(self) -> bool:
        """
        Say whether a question is on screen; the title is the state here,
        there is no separate flag to keep in step with it

        :returns: True while a question is waiting for an answer
        """
        return self.title is not None

    def _fonts(self, panel_w: int) -> tuple[pg.font.Font, pg.font.Font, pg.font.Font]:
        """
        Fetch the three fonts for the current panel width, rebuilt only when
        that width changes so no font is loaded per frame

        :param panel_w: panel width in pixels the fonts are sized from
        :returns: title, sub-line and button fonts
        """
        return cast(tuple[pg.font.Font, pg.font.Font, pg.font.Font],
                    fonts_for_width(self._font_cache, panel_w, self._build_fonts))

    def _build_fonts(self, panel_w: float) -> tuple[pg.font.Font, pg.font.Font, pg.font.Font]:
        """
        Size the title, sub-line and button fonts from the panel width, each
        with a floor so the box stays readable in a small window

        :param panel_w: panel width in pixels
        :returns: title, sub-line and button fonts
        """
        return (
            get_display_font(max(int(panel_w * 0.07), 22)),
            get_font(max(int(panel_w * 0.032), 13), bold=False),
            get_font(max(int(panel_w * 0.034), 13), bold=True),
        )

    def draw(self) -> None:
        """
        Paint the question: the shared shell, an optional emoji tile, the
        title, the wrapped explanation and the button row, in a panel sized to
        exactly what is there. Drawing is also what fixes the button rects
        that clicks are tested against, so a button left out of the row cannot
        be clicked either
        """
        if not self.is_visible() or self.rect.width <= 0:
            self.button_rects = {}
            return
        pad = self.padding
        panel_w = min(self.rect.width, MODAL_MAX_WIDTH)
        inner_w = panel_w - 2 * pad
        title_font, sub_font, button_font = self._fonts(panel_w)
        btn_h = max(int(panel_w * 0.11), 40)

        title_surf = fit_text_to_rect(
            title_font.render(cast(str, self.title).upper(), True, Colors.text),
            pg.Rect(0, 0, inner_w, title_font.get_height()))
        sub_lines = wrap_words(self.sub, sub_font, inner_w, SUB_MAX_LINES) if self.sub else []
        line_h = sub_font.get_linesize()

        icon_side = max(int(panel_w * 0.12), 40) if self.emoji else 0
        gap_icon = max(int(panel_w * 0.03), 12) if self.emoji else 0
        gap_title = TITLE_SUB_GAP if sub_lines else 0
        block_h = (icon_side + gap_icon + title_surf.get_height()
                   + (gap_title + line_h * len(sub_lines) if sub_lines else 0))
        panel_h = MODAL_RAIL + pad + block_h + max(int(panel_w * 0.05), 18) + btn_h + pad
        panel = pg.Rect(0, 0, panel_w, panel_h)
        panel.center = self.rect.center
        self._panel = panel

        self.draw_shell("loss" if self.danger else None, panel)
        content = self.content_rect(panel)
        y = content.y
        if self.emoji:
            tile = pg.Rect(content.centerx - icon_side // 2, y, icon_side, icon_side)
            fill: str | pg.Color = Colors.surface_hover
            border: str | pg.Color = Colors.border
            if self.danger:
                fill = pg.Color(Colors.loss).lerp(pg.Color(Colors.surface_hover), 0.84)
                border = pg.Color(Colors.loss).lerp(pg.Color(Colors.surface_raised), 0.6)
            self.window.blit(cut_rect_surface(tile.size, TITLE_TILE_CUT, fill,
                                              border=border, border_width=1,
                                              corners=("tr", "bl")), tile.topleft)
            blit_emoji(self.window, self.emoji, tile.center, int(icon_side * 0.62))
            y += icon_side + gap_icon
        self.window.blit(title_surf, (content.centerx - title_surf.get_width() / 2, y))
        y += title_surf.get_height() + gap_title
        for line in sub_lines:
            surf = sub_font.render(line, True, Colors.text_dim)
            self.window.blit(surf, (content.centerx - surf.get_width() / 2, y))
            y += line_h

        row = pg.Rect(content.x, content.bottom - btn_h, content.width, btn_h)
        buttons = [(self.no_label, "no")] if self.no_label else []
        buttons.append((self.yes_label, "yes"))
        if self.on_extra is not None:
            buttons.append((self.extra_label, "extra"))
        self.button_rects = draw_button_row(
            self.window, row, buttons, button_font, pad, primary_keys={"yes"}, cut=True)

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """
        Turn a click on one of the buttons into the answer it stands for. The
        box closes before the callback runs, so an answer that opens another
        modal finds this one already out of the way

        :param pos: click position in window pixels
        :returns: True when a button took the click
        """
        if not self.is_visible():
            return False
        callbacks = {"yes": self.on_yes, "no": self.on_no, "extra": self.on_extra}
        for key, br in self.button_rects.items():
            if br.collidepoint(pos):
                callback = callbacks.get(key)
                self.hide()
                if callback is not None:
                    callback()
                return True
        return False
