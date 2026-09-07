"""OptionsView + the layout-agnostic settings rows (v2.9.0: Options folded from a
global modal into a menu sub-view). The row widgets (toggle/segmented/notch/path/
text -- the old slider is NotchRow now, SwatchRow retired) route clicks to their
getter/setter exactly as before; the view itself hosts them in an OptionsBody
ScrollHost inside the menu's subview_rect and replaces the old modal close-gate
with an on-exit gate that fires on view exit AND app quit, never veto-blocking
navigation."""

import os
import threading
from types import SimpleNamespace

import pygame as pg
import pytest

from tests.conftest import pygame_display
from chessshootout import paths
from chessshootout.infra import env
from chessshootout.frontend import settings as settings_module
from chessshootout.frontend.menu import options_rows
from chessshootout.frontend.menu.options_rows import (
    OptionsBody, PathRow, TextRow, ToggleRow, SegmentedRow, NotchRow,
    RevealRow, ActionRow, CARD_PAD_X, SCROLLBAR_RESERVE,
    ACTION_MIN_LABEL_W, ACTION_STATUS_GAP, CONTROL_GAP,
    NOTCH_COUNT, NOTCH_CELL_W, NOTCH_GAP, Fonts, _elide_left, _fitting_ellipsis,
)
from chessshootout.frontend.settings import (
    SERVER_PROBE_BUTTON_IDLE, SERVER_PROBE_BUTTON_PENDING, SERVER_SWITCH_BLOCKED_MESSAGE,
)
from chessshootout.online.transport import HEALTHZ_TIMEOUT_SECONDS
from chessshootout.frontend.visual.colors import Colors
from chessshootout.frontend.visual.fonts import get_font, get_mono_font
from chessshootout.server.protocol import PROTOCOL_VERSION
from tests.helpers import assert_pixel_color, make_app


_pygame_init = pygame_display(1200, 900)

_SERVER_TARGET_VARS = ("CHESS_SERVER_MODE", "CHESS_CUSTOM_SERVER_ADDR", "CHESS_SERVER_ADDR")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep CHESS_DATA_DIR unset for this module's launch-mode / data-folder tests.
    The root conftest's _isolate_env_file autouse already pins env._ENV_PATH to a
    temp .env, so no need to re-patch it here. The server-target trio needs the
    same treatment: env.set_server_mode/set_custom_server_addr write os.environ
    directly (not via monkeypatch), so a test that switches to Official would
    otherwise leak the resolved production address into later tests."""
    monkeypatch.delenv("CHESS_DATA_DIR", raising=False)
    for var in _SERVER_TARGET_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    os.environ.pop("CHESS_DATA_DIR", None)
    for var in _SERVER_TARGET_VARS:
        os.environ.pop(var, None)


def _fonts():
    return Fonts(get_font(14, bold=True), get_font(11), get_font(10, bold=True),
                 get_mono_font(12), get_font(13, bold=True))


def _draw_row(row, rect=pg.Rect(40, 40, 420, 56)):
    win = pg.display.get_surface()
    win.fill((0, 0, 0))
    row.draw(win, rect, _fonts())
    return row


def _draw_then_click_confirm(app, key):
    app.draw_frame()
    app.confirm_modal.handle_click(app.confirm_modal.button_rects[key].center)


@pytest.fixture
def app():
    application = make_app(1200, 900)
    application.draw_frame()
    return application


@pytest.fixture
def view(app):
    return app.menu.views["options"]


# --- row widgets (layout-agnostic, no app needed) --------------------------

def test_toggle_row_flips_value():
    state = {"on": False}
    row = ToggleRow("Reduce motion", "calm", lambda: state["on"],
                    lambda v: state.update(on=v))
    _draw_row(row)
    assert row.handle_click(row._ctl.center) is True
    assert state["on"] is True


def test_toggle_knob_animates_toward_target():
    """After a flip the knob eases toward the new state across frames rather than
    snapping instantly."""
    state = {"on": False}
    row = ToggleRow("Reduce motion", "calm", lambda: state["on"],
                    lambda v: state.update(on=v))
    _draw_row(row)
    assert row._pos == 0.0
    state["on"] = True
    _draw_row(row)
    assert 0.0 < row._pos < 1.0
    for _ in range(20):
        _draw_row(row)
    assert row._pos == 1.0


def test_segmented_row_selects_option():
    chosen = {"v": "a"}
    row = SegmentedRow("Intensity", "", [("A", "a"), ("B", "b"), ("C", "c")],
                       lambda: chosen["v"], lambda k: chosen.update(v=k))
    _draw_row(row)
    assert row.handle_click(row._rects["c"].center) is True
    assert chosen["v"] == "c"


def test_notch_row_clicks_a_cell_to_set_that_level():
    """cp2: the continuous slider became ~10 discrete notch cells. Clicking cell N
    (1-indexed) sets the value to N / NOTCH_COUNT."""
    val = {"v": 0.0}
    row = NotchRow("Volume", "", lambda: val["v"], lambda v: val.update(v=v))
    _draw_row(row)
    step = NOTCH_CELL_W + NOTCH_GAP

    def cell_center(i):
        return (row._band.x + i * step + NOTCH_CELL_W // 2, row._band.centery)

    assert row.handle_click(cell_center(7)) is True
    assert val["v"] == pytest.approx(0.8)
    row.handle_click(cell_center(0))
    assert val["v"] == pytest.approx(0.1)
    row.handle_click(cell_center(NOTCH_COUNT - 1))
    assert val["v"] == pytest.approx(1.0)


def test_notch_row_first_cell_reclick_toggles_to_zero():
    """The first notch doubles as an off switch: clicking it at 10% drops the value
    to 0%, clicking again restores 10%. Other cells never toggle to zero."""
    val = {"v": 0.0}
    row = NotchRow("Volume", "", lambda: val["v"], lambda v: val.update(v=v))
    _draw_row(row)
    step = NOTCH_CELL_W + NOTCH_GAP

    def cell_center(i):
        return (row._band.x + i * step + NOTCH_CELL_W // 2, row._band.centery)

    row.handle_click(cell_center(0))
    assert val["v"] == pytest.approx(0.1)
    row.handle_click(cell_center(0))
    assert val["v"] == pytest.approx(0.0)
    row.handle_click(cell_center(0))
    assert val["v"] == pytest.approx(0.1)
    row.handle_click(cell_center(4))
    assert val["v"] == pytest.approx(0.5)
    row.handle_click(cell_center(4))
    assert val["v"] == pytest.approx(0.5)


def test_notch_band_position_is_stable_across_readout_widths():
    """The notch band anchors to a fixed-width readout slot sized for '100%', so
    switching between 0%, 50%, and 100% never shifts the cells horizontally."""
    val = {"v": 0.0}
    row = NotchRow("Volume", "", lambda: val["v"], lambda v: val.update(v=v))
    positions = []
    for v in (0.0, 0.5, 1.0):
        val["v"] = v
        _draw_row(row)
        positions.append(row._band.x)
    assert positions[0] == positions[1] == positions[2]


def test_notch_row_click_fires_tick_and_debounced_write():
    """Clicking a notch cell plays the tick AND schedules the debounced env write
    (both preserved from the old slider's on_tick / on_release wiring)."""
    ticks, writes = [], []
    row = NotchRow("Volume", "", lambda: 0.5, lambda v: None,
                   on_tick=lambda: ticks.append(1), on_release=lambda: writes.append(1))
    _draw_row(row)
    assert row.handle_click((row._band.centerx, row._band.centery)) is True
    assert ticks == [1]
    assert writes == [1]


def test_notch_row_fill_paints_accent_cells_up_to_value():
    """The filled notch cells wear the accent; empties are sunken border wells."""
    row = NotchRow("Volume", "", lambda: 0.8, lambda v: None)
    win = pg.display.get_surface()
    win.fill((0, 0, 0))
    row.draw(win, pg.Rect(40, 40, 460, 56), _fonts())
    step = row._band.width / NOTCH_COUNT
    filled_x = int(row._band.x + step * 0.5)
    empty_x = int(row._band.x + step * 9.5)
    assert_pixel_color(win, filled_x, row._band.centery, Colors.accent, tol=24)
    empty = win.get_at((empty_x, row._band.centery))[:3]
    assert empty != tuple(pg.Color(Colors.accent))[:3]


# SwatchRow (the board-theme picker) was retired in v2.9.0: theme/customization
# moves to the Armory battle pass, so it left the settings UI entirely (the
# CHESS_THEME env plumbing was removed along with it, its last consumer). The
# two swatch draw/click tests were removed with the widget; the Display
# section's Launch mode row replaces it.


def test_left_elide_prepends_ellipsis_and_preserves_the_exact_tail():
    """A path too wide for its field is truncated from the LEFT with a leading
    ellipsis; the visible tail is an exact suffix of the original and the whole
    thing fits the budget. A short path is returned untouched."""
    font = get_mono_font(13)
    long_path = "/a/very/long/synthetic/path/for/testing/left/elide"
    budget = 150
    out = _elide_left(font, long_path, budget)
    ell = _fitting_ellipsis(font)
    assert out.startswith(ell)
    assert out != long_path
    tail = out[len(ell):]
    assert long_path.endswith(tail) and tail != ""
    assert font.size(out)[0] <= budget
    assert _elide_left(font, "/tmp/x", 400) == "/tmp/x"


def test_fitting_ellipsis_prefers_the_single_glyph_when_the_font_has_it():
    """The bundled mono font carries U+2026; the helper uses it. (If a font lacked
    it — some miss glyphs like the arrow — the helper falls back to three dots.)"""
    assert _fitting_ellipsis(get_mono_font(13)) == "…"


def _path_row(getter=lambda: "/tmp/x", on_change=lambda: None, on_reset=lambda: None):
    return PathRow("Games folder", "where games live", pg.display.get_surface(),
                   getter, on_change, on_reset)


def test_pathrow_buttons_route_callbacks():
    fired = {"change": 0, "reset": 0}
    row = _path_row(on_change=lambda: fired.update(change=fired["change"] + 1),
                    on_reset=lambda: fired.update(reset=fired["reset"] + 1))
    _draw_row(row, pg.Rect(40, 40, 420, 120))
    assert row.handle_click(row._change_rect.center) is True
    assert row.handle_click(row._reset_rect.center) is True
    assert fired == {"change": 1, "reset": 1}


def test_pathrow_field_click_focuses_input():
    row = _path_row()
    _draw_row(row, pg.Rect(40, 40, 420, 120))
    assert row.handle_click(row._field_rect.center) is True
    assert row.input.focused is True


def test_pathrow_current_text_strips_and_reflects_input():
    row = _path_row()
    _draw_row(row, pg.Rect(40, 40, 420, 120))
    row.input.text = "  /home/me/chess  "
    assert row.current_text() == "/home/me/chess"


def test_pathrow_handle_key_types_into_focused_field():
    row = _path_row()
    _draw_row(row, pg.Rect(40, 40, 420, 120))
    row.input.text = ""
    row.input.focused = True
    ev = pg.event.Event(pg.KEYDOWN, key=pg.K_z, mod=0, unicode="z")
    assert row.handle_key(ev) is True
    assert row.current_text() == "z"


def test_pathrow_cancel_edit_restores_the_stored_folder():
    # PathRow holds dirty text exactly like TextRow does -- current_text() reads
    # straight out of the TextInput -- and commit_options_exit() feeds that text
    # to the data-folder validator. Without its own cancel it had no way to give
    # the typed path back, so it needs the same restore TextRow got.
    row = _path_row(getter=lambda: "/tmp/x")
    _draw_row(row, pg.Rect(40, 40, 420, 120))
    row.input.focused = True
    row.input.text = "/half/typed"
    assert row.cancel_edit() is True
    assert row.input.focused is False
    assert row.current_text() == "/tmp/x"


def test_pathrow_with_nothing_focused_declines_the_escape():
    # the base _Row declares cancel_edit(); every row answers it, and a row that
    # is not being edited must decline so the Esc falls through to the sub-view.
    row = _path_row()
    _draw_row(row, pg.Rect(40, 40, 420, 120))
    assert row.input.focused is False
    assert row.cancel_edit() is False
    assert ToggleRow("Reduce motion", "calm", lambda: False,
                     lambda v: None).cancel_edit() is False


def test_text_row_reports_typed_value():
    row = TextRow("Server", "where online connects", pg.display.get_surface(),
                  lambda: "localhost:8000", placeholder="host or host:port")
    _draw_row(row)
    assert row.current_text() == "localhost:8000"
    row.input.focused = True
    row.input.text = "chess.example.com:9000"
    assert row.current_text() == "chess.example.com:9000"


def _type(row, text):
    row.input.focused = True
    for ch in text:
        row.handle_key(pg.event.Event(pg.KEYDOWN, key=ord(ch), mod=0, unicode=ch))


def test_enter_commits_the_typed_value_instead_of_snapping_it_back():
    # The row repaints from its getter whenever the field is unfocused, and
    # RETURN unfocuses it -- so before the commit hook existed, pressing enter
    # handed the typed address straight back to the getter and the next frame
    # painted the OLD value over it. The user's edit was gone before anything
    # could persist it, and the exit commit then wrote back what was already
    # stored.
    stored = ["localhost:8000"]
    row = TextRow("Server", "where online connects", pg.display.get_surface(),
                  lambda: stored[0], on_commit=lambda v: stored.__setitem__(0, v))
    _draw_row(row)
    row.input.text = ""
    _type(row, "example.com")
    row.handle_key(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, mod=0, unicode="\r"))
    assert stored[0] == "example.com", "enter is what saves the address"
    _draw_row(row)
    assert row.current_text() == "example.com", "and the field keeps showing it"


def test_an_unfocused_edit_survives_until_the_row_is_committed():
    # Clicking away parks the edit instead of destroying it: the exit commit is
    # still the backstop for anyone who types and leaves without pressing enter.
    row = TextRow("Server", "where online connects", pg.display.get_surface(),
                  lambda: "localhost:8000")
    _draw_row(row)
    row.input.text = ""
    _type(row, "kept.example")
    row.input.focused = False
    _draw_row(row)
    assert row.current_text() == "kept.example"


def test_escape_cancels_the_edit_and_restores_the_stored_value():
    # Escape never reaches the row's key handler: input_router intercepts it
    # and calls screen.escape(), so a cancel implemented in handle_key would be
    # dead code that only a unit test could reach. It has to live on the view's
    # escape(), which is the first link in the menu's Esc chain -- cancel the
    # field, THEN fall through to leaving the sub-view on a second press.
    row = TextRow("Server", "where online connects", pg.display.get_surface(),
                  lambda: "localhost:8000")
    body = OptionsBody()
    body.set_sections([("Online", [row])])
    body.draw(pg.display.get_surface(), pg.Rect(40, 40, 460, 240), _fonts())
    row.input.text = ""
    _type(row, "typo.example")
    assert body.cancel_focused_edit() is True, "the focused edit consumes the escape"
    body.draw(pg.display.get_surface(), pg.Rect(40, 40, 460, 240), _fonts())
    assert row.current_text() == "localhost:8000"
    assert row.input.focused is False
    assert body.cancel_focused_edit() is False, \
        "with nothing being edited escape falls through to the sub-view"


def test_text_row_ignores_keys_while_it_is_not_focused():
    # the row only owns the keyboard while its own field is focused; otherwise it
    # must decline so OptionsBody.handle_key keeps walking to the next row.
    row = TextRow("Server", "where online connects", pg.display.get_surface(),
                  lambda: "localhost:8000")
    _draw_row(row)
    assert row.input.focused is False
    assert row.handle_key(pg.event.Event(pg.KEYDOWN, key=pg.K_z, mod=0, unicode="z")) is False
    assert row.current_text() == "localhost:8000", "and nothing was typed into it"


def test_a_row_with_no_edit_still_follows_its_getter():
    value = ["localhost:8000"]
    row = TextRow("Server", "where online connects", pg.display.get_surface(),
                  lambda: value[0])
    _draw_row(row)
    value[0] = "changed.elsewhere"
    _draw_row(row)
    assert row.current_text() == "changed.elsewhere"


def test_action_row_elides_a_long_status_instead_of_crushing_the_label():
    """A verdict like the protocol-mismatch line can outgrow a narrow row; the
    status must clamp to the space right of a reserved label width and elide,
    never pushing control_left past the title area (which collapsed the row
    title to a 1px sliver before)."""
    long_status = "Protocol mismatch: server v3 ≠ client v4 on a narrow window"
    row = ActionRow("Connection", "Check the server answers", lambda: "Test",
                    lambda: None, lambda: ("warn", long_status))
    win = pg.display.get_surface()
    win.fill((0, 0, 0))
    rect = pg.Rect(40, 40, 360, 56)
    fonts = _fonts()
    left = row._draw_control(win, rect, fonts)
    assert left >= rect.x + ACTION_MIN_LABEL_W, \
        "the label keeps its reserved width no matter how long the status runs"
    avail = row._button_rect.x - ACTION_STATUS_GAP \
        - (rect.x + ACTION_MIN_LABEL_W + CONTROL_GAP)
    assert fonts.value.size(long_status)[0] > avail, "the status really overflows"
    _key, shown = row._status_cache
    assert shown.get_width() <= avail, "what gets drawn is the elided fit"


def test_action_row_short_status_still_sits_flush_against_the_button():
    row = ActionRow("Connection", "desc", lambda: "Test",
                    lambda: None, lambda: ("ok", "OK · 12 ms"))
    win = pg.display.get_surface()
    win.fill((0, 0, 0))
    fonts = _fonts()
    left = row._draw_control(win, pg.Rect(40, 40, 520, 56), fonts)
    _key, shown = row._status_cache
    assert shown.get_width() == fonts.value.size("OK · 12 ms")[0], "no elision when it fits"
    assert left == row._button_rect.x - ACTION_STATUS_GAP - shown.get_width()


# --- OptionsBody scrolling (standalone, no view/app needed) ----------------

def test_body_scroll_clamps():
    body = OptionsBody()
    rows = [ToggleRow(f"opt {i}", "desc", lambda: False, lambda v: None) for i in range(12)]
    body.set_sections([("Many", rows)])
    surf = pg.display.get_surface()
    rect = pg.Rect(100, 60, 460, 240)
    body.draw(surf, rect, _fonts())
    body.handle_scroll((rect.centerx, rect.centery), -5)
    assert body.scroll_offset > 0
    body.handle_scroll((rect.centerx, rect.centery), 999)
    assert body.scroll_offset == 0


# --- OptionsView hosting ----------------------------------------------------

def test_enter_builds_six_sections_dropping_profile(app, view):
    app.menu.goto_view("options")
    labels = [label for label, _ in view.body.sections]
    assert labels == ["Audio", "Display", "Focus mode", "Game", "Online", "Performance"]
    assert "Profile" not in labels
    assert "Appearance" not in labels
    online_rows = dict(view.body.sections)["Online"]
    assert [type(r) for r in online_rows] == [SegmentedRow, RevealRow, ActionRow, ToggleRow]
    assert online_rows[0].title == "Server"
    assert [key for _, key in online_rows[0].options] == ["official", "custom"]
    assert online_rows[1].inner is app.settings._custom_server_row
    assert online_rows[2].title == "Connection"
    assert online_rows[3].title == "Hide opponent's marks"


def test_section_card_paints_surface_fill_and_chamfers_top_right(app):
    """cp2: each section is its own cut-corner card (Colors.surface fill, 1px border,
    top-right corner chamfered) rather than one big surface_raised panel."""
    app.menu.goto_view("options")
    view = app.menu.views["options"]
    app.window.fill((0, 0, 0))
    view.draw(app.window, app.menu._menu_layout)
    card = view.body._card_rects[0]
    assert_pixel_color(app.window, card.x + 4, card.bottom - 4, Colors.surface, tol=10)
    assert_pixel_color(app.window, card.x + 3, card.y + 3, Colors.surface, tol=10)
    corner = app.window.get_at((card.right - 2, card.y + 2))[:3]
    assert corner != tuple(pg.Color(Colors.surface))[:3], "top-right corner is chamfered away"


def test_one_card_per_section(app):
    app.menu.goto_view("options")
    view = app.menu.views["options"]
    app.window.fill((0, 0, 0))
    view.draw(app.window, app.menu._menu_layout)
    assert len(view.body._card_rects) == len(view.body.sections) == 6


def test_value_cells_hit_test_selects(app):
    """Default time / increment render as small square cut-corner value cells; each
    cell is its own hit target keyed to the option value."""
    chosen = {"v": "10"}
    row = SegmentedRow("Default time", "", [(v, v) for v in ("1", "3", "5", "10", "∞")],
                       lambda: chosen["v"], lambda k: chosen.update(v=k),
                       mono=True, variant="cells")
    _draw_row(row, pg.Rect(40, 40, 460, 60))
    assert row.handle_click(row._rects["∞"].center) is True
    assert chosen["v"] == "∞"


def test_focus_chips_hit_test_selects(app):
    chosen = {"v": "line"}
    row = SegmentedRow("Show in focus", "", [("Nothing", "nothing"), ("Time Line", "line"),
                                             ("Full strips", "strips")],
                       lambda: chosen["v"], lambda k: chosen.update(v=k), variant="chips")
    _draw_row(row, pg.Rect(40, 40, 460, 60))
    assert row.handle_click(row._rects["strips"].center) is True
    assert chosen["v"] == "strips"


def test_launch_mode_row_lives_in_display_and_writes_the_env(app, monkeypatch):
    """The Display section's Launch mode chips write CHESS_LAUNCH_MODE straight
    through env.set_launch_mode; it applies on next launch, not live."""
    monkeypatch.delenv("CHESS_LAUNCH_MODE", raising=False)
    app.menu.goto_view("options")
    rows = dict(app.menu.views["options"].body.sections)["Display"]
    launch_row = next(r for r in rows if r.title == "Launch mode")
    _draw_row(launch_row, pg.Rect(40, 40, 520, 60))
    assert launch_row.handle_click(launch_row._rects["fullscreen"].center) is True
    assert env.get_launch_mode() == "fullscreen"
    assert launch_row.handle_click(launch_row._rects["windowed"].center) is True
    assert env.get_launch_mode() == "windowed"


def test_sections_paint_nonblank_pixels(app):
    app.menu.goto_view("options")
    view = app.menu.views["options"]
    app.window.fill((0, 0, 0))
    view.draw(app.window, app.menu._menu_layout)
    painted = app.window.subsurface(view._body_rect)
    assert pg.image.tobytes(painted, "RGB") != bytes(
        painted.get_width() * painted.get_height() * 3)


def test_esc_from_options_returns_to_play(app):
    app.menu.goto_view("options")
    result = app.menu.escape()
    assert result is True
    assert app.menu._active_view == "play"


def test_click_routes_to_options_body_not_play_view(app, monkeypatch):
    """With options active, clicks route to it and never reach play_view."""
    app.menu.goto_view("options")
    app.draw_frame()
    received = []
    monkeypatch.setattr(app.menu.play_view, "handle_click", lambda pos: received.append(pos))
    app.input_router.mouse_left_clicked((10, 10))
    assert received == []
    assert app.menu._active_view == "options"


def test_auto_queen_row_lives_in_game_and_writes_env(app, monkeypatch):
    """The Game section's Auto-queen toggle writes CHESS_AUTO_QUEEN straight through
    env.get/set_auto_queen; the promotion picker is bypassed but the skill check
    still fires (enforced in the board tests)."""
    monkeypatch.delenv("CHESS_AUTO_QUEEN", raising=False)
    app.menu.goto_view("options")
    rows = dict(app.menu.views["options"].body.sections)["Game"]
    auto_row = next(r for r in rows if r.title == "Auto-queen")
    assert auto_row.getter() is False
    _draw_row(auto_row, pg.Rect(40, 40, 520, 60))
    assert auto_row.handle_click(auto_row._ctl.center) is True
    assert env.get_auto_queen() is True
    assert auto_row.handle_click(auto_row._ctl.center) is True
    assert env.get_auto_queen() is False


def test_mute_toggle_row_wired_to_sound_manager(app):
    from unittest.mock import MagicMock
    app.sound_manager = MagicMock(enabled=True)
    app.menu.goto_view("options")
    rows = dict(app.menu.views["options"].body.sections)["Audio"]
    mute_row = next(r for r in rows if r.title == "Mute all sound")
    assert mute_row.getter() is False
    mute_row.setter(True)
    app.sound_manager.set_enabled.assert_called_once_with(False)


def test_handle_key_routes_to_the_focused_pathrow(app):
    app.menu.goto_view("options")
    app.draw_frame()
    row = app.settings._data_folder_row
    row.input.text = ""
    row.input.focused = True
    ev = pg.event.Event(pg.KEYDOWN, key=pg.K_q, mod=0, unicode="q")
    assert app.menu.views["options"].handle_key(ev) is True
    assert row.current_text() == "q"


def test_directory_browser_esc_closes_browser_not_the_options_view(app):
    """The directory browser opens from inside Options and is a GLOBAL modal
    layered above the view (per the modal-registry ordering): Esc peels it
    first and leaves the options view underneath still active."""
    app.menu.goto_view("options")
    app.directory_browser.show(str(paths.get_data_dir()), lambda p: None)
    assert app.directory_browser.is_visible() is True

    app.input_router._handle_escape()
    assert app.directory_browser.is_visible() is False
    assert app.menu._active_view == "options"

    app.input_router._handle_escape()
    assert app.menu._active_view == "play"


def test_no_options_modal_survives_on_the_frontend(app):
    assert not hasattr(app, "options_modal")


# --- on-exit gate: server address, defaults sync, deferred writes ----------

def test_exit_commits_a_pending_custom_address(app):
    app.menu.goto_view("options")
    app.settings._custom_server_row.input.text = "chess.example.com:9000"
    app.menu.goto_view("play")
    assert env.get_custom_server_addr() == "chess.example.com:9000"
    assert env.get_server_addr() == "chess.example.com:9000"


def test_exit_syncs_default_time_to_the_play_picker(monkeypatch):
    app = make_app()
    app.menu.play_view.selected_time_minutes = 5
    app.menu.play_view.selected_increment_seconds = 2
    monkeypatch.setenv("CHESS_DEFAULT_TC", "15")
    monkeypatch.setenv("CHESS_DEFAULT_INCREMENT", "10")
    app.menu.goto_view("options")
    app.menu.goto_view("play")
    assert app.menu.play_view.selected_time_minutes == 15
    assert app.menu.play_view.selected_increment_seconds == 10


def test_exit_force_flushes_a_pending_deferred_env_write(app):
    fired = []
    app.settings._defer_env_write("master_volume", lambda: fired.append(1))
    app.menu.goto_view("options")
    app.menu.goto_view("play")
    assert fired == [1]


def test_on_app_exit_commits_pending_options_even_without_navigating_away(app):
    """The gate must also fire at app quit (MenuScreen.on_app_exit), covering
    the case where the user quits while Options is still the active view and
    never triggers its own exit() through goto_view."""
    app.menu.goto_view("options")
    app.settings._custom_server_row.input.text = "quit-addr.example.com:9000"
    app.menu.on_app_exit()
    assert env.get_server_addr() == "quit-addr.example.com:9000"


def test_invalid_data_folder_on_exit_toasts_and_does_not_veto_navigation(app):
    app.menu.goto_view("options")
    app.settings._data_folder_row.input.text = "/definitely/not/writable/anywhere"
    app.menu.goto_view("play")
    assert app.toast.message == "That folder isn't writable"
    assert app.menu._active_view == "play"


# --- CROSSCHECK 7: games-folder move flow survives the exit gate -----------

def test_valid_changed_folder_with_games_on_exit_prompts_move_confirm(tmp_path, monkeypatch):
    cur = tmp_path / "cur"
    (cur / "games").mkdir(parents=True)
    (cur / "games" / "g.pgn").write_text("x", encoding="utf-8")
    monkeypatch.setenv("CHESS_DATA_DIR", str(cur))
    app = make_app()
    app.menu.goto_view("options")
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    app.settings._data_folder_row.input.text = str(new_dir)

    app.menu.goto_view("play")

    assert app.confirm_modal.is_visible() is True
    _draw_then_click_confirm(app, "yes")
    assert (new_dir / "games" / "g.pgn").read_text(encoding="utf-8") == "x"
    assert not (cur / "games").exists()
    assert str(paths.get_data_dir()) == str(new_dir)


def test_changed_folder_dont_move_still_commits_the_new_dir(tmp_path, monkeypatch):
    cur = tmp_path / "cur"
    (cur / "games").mkdir(parents=True)
    (cur / "games" / "g.pgn").write_text("x", encoding="utf-8")
    monkeypatch.setenv("CHESS_DATA_DIR", str(cur))
    app = make_app()
    app.menu.goto_view("options")
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    app.settings._data_folder_row.input.text = str(new_dir)

    app.menu.goto_view("play")
    _draw_then_click_confirm(app, "no")

    assert (cur / "games" / "g.pgn").exists()
    assert not (new_dir / "games" / "g.pgn").exists()
    assert str(paths.get_data_dir()) == str(new_dir)


def test_escape_in_the_custom_server_field_cancels_before_it_leaves_options(app, monkeypatch):
    """The whole Esc chain, end to end, on a real app.

    This is the shape that matters: the router hands Esc to screen.escape(),
    the menu asks the active view first, and only if the view declines does it
    fall back to leaving the sub-view -- which runs commit_options_exit(). So a
    half-typed address must be cancelled by the FIRST escape and must never
    reach the commit, or escape would silently save the thing it looks like it
    is discarding. Since #89 the row commits through env.set_custom_server_addr
    and lives inside a RevealRow, whose cancel_edit delegates while the row is
    open (custom mode is the source-run default).
    """
    saved = []
    monkeypatch.setattr(env, "set_custom_server_addr", lambda v: saved.append(v))
    app.menu.goto_view("options")
    app.draw_frame()
    row = app.settings._custom_server_row
    row.input.focused = True
    row.input.text = "half-typed.example"

    assert app.screen.escape() is True
    assert app.menu._active_view == "options", "the first escape stays in options"
    assert saved == [], "a cancelled edit is never committed"
    app.draw_frame()
    assert row.current_text() == env.get_custom_server_addr()

    assert app.screen.escape() is True
    assert app.menu._active_view == "play", "a second escape leaves the sub-view"


def test_escape_in_the_games_folder_field_cancels_before_the_exit_validation(app, monkeypatch):
    """The same Esc chain over the OTHER editable row, which had no cancel of its
    own. PathRow keeps the typed path in its TextInput and commit_options_exit()
    reads it back through current_text(), so before this the first Esc fell
    straight through to leaving the sub-view and handed the half-typed folder to
    _validate_data_folder_on_exit -- a bogus "isn't writable" toast on a path the
    user was still typing, or a real data-dir move if the fragment happened to
    name a writable directory.
    """
    moved = []
    monkeypatch.setattr(app.settings, "_apply_data_folder_change", moved.append)
    app.menu.goto_view("options")
    app.draw_frame()
    row = app.settings._data_folder_row
    stored = str(paths.get_data_dir())
    row.input.focused = True
    row.input.text = "/half/typed/folder"

    assert app.screen.escape() is True
    assert app.menu._active_view == "options", "the first escape stays in options"
    assert row.input.focused is False
    assert row.current_text() == stored, "the field snaps back to the stored folder"

    assert app.screen.escape() is True
    assert app.menu._active_view == "play", "a second escape leaves the sub-view"
    assert moved == [], "the cancelled path never reaches the data-dir move"
    assert app.toast.message != "That folder isn't writable", \
        "nor the validator that would have complained about it"


# --- #89 Online section: server mode + custom-address reveal ----------------

def _online_rows(app, view):
    app.menu.goto_view("options")
    app.draw_frame()
    return dict(view.body.sections)["Online"]


def _settle(app, view, frames=60):
    for _ in range(frames):
        view.draw(app.window, app.menu._menu_layout)


def test_server_mode_row_switches_between_official_and_custom(app, view):
    """The Server chips write CHESS_SERVER_MODE through the controller AND
    rewrite the resolved active address, so transport reads the right host the
    moment the chip lands."""
    seg = _online_rows(app, view)[0]
    assert env.get_server_mode() == "custom", "source runs default to custom"
    _draw_row(seg, pg.Rect(40, 40, 520, 60))
    assert seg.handle_click(seg._rects["official"].center) is True
    assert env.get_server_mode() == "official"
    assert env.get_server_addr() == env.official_server_addr()
    _draw_row(seg, pg.Rect(40, 40, 520, 60))
    assert seg.handle_click(seg._rects["custom"].center) is True
    assert env.get_server_mode() == "custom"
    assert env.get_server_addr() == env.get_custom_server_addr()


def test_official_mode_always_resolves_to_the_official_address(app):
    """A stored custom address never bleeds into Official mode; switching back
    restores it."""
    env.set_custom_server_addr("10.0.0.5:8000")
    app.settings._apply_server_mode("official")
    assert env.get_server_addr() == env.official_server_addr()
    app.settings._apply_server_mode("custom")
    assert env.get_server_addr() == "10.0.0.5:8000"


def test_custom_row_measures_zero_height_in_official_mode(app, view, monkeypatch):
    monkeypatch.setenv("CHESS_SERVER_MODE", "official")
    reveal = _online_rows(app, view)[1]
    assert reveal._t == 0.0
    assert reveal.height(view._fonts()) == 0


def test_entering_options_in_custom_mode_starts_fully_open_without_animating(app, view):
    """RevealRow seeds _t to the settled value at construction, and OptionsView
    rebuilds the rows on enter — so opening Options never plays the reveal."""
    reveal = _online_rows(app, view)[1]
    assert reveal._t == 1.0
    fonts = view._fonts()
    assert reveal.height(fonts) == reveal.inner.height(fonts)


def test_switching_to_custom_animates_the_row_open_over_several_frames(app, view, monkeypatch):
    monkeypatch.setenv("CHESS_SERVER_MODE", "official")
    reveal = _online_rows(app, view)[1]
    fonts = view._fonts()
    full = reveal.inner.height(fonts)
    assert reveal.height(fonts) == 0
    monkeypatch.setenv("CHESS_SERVER_MODE", "custom")
    heights = []
    for _ in range(60):
        view.draw(app.window, app.menu._menu_layout)
        heights.append(reveal.height(fonts))
    assert heights == sorted(heights), "the reveal only ever grows"
    assert 0 < heights[0] < full, "the first frame is partial, not a snap"
    assert len({h for h in heights if 0 < h < full}) >= 4, \
        "several distinct in-between frames"
    assert heights[-1] == full, "and it settles at the full row height"
    assert reveal._t == 1.0


def test_switching_to_official_animates_the_row_closed_and_settles_at_zero(app, view):
    reveal = _online_rows(app, view)[1]
    fonts = view._fonts()
    full = reveal.inner.height(fonts)
    assert reveal.height(fonts) == full
    app.settings._apply_server_mode("official")
    heights = []
    for _ in range(60):
        view.draw(app.window, app.menu._menu_layout)
        heights.append(reveal.height(fonts))
    assert heights == sorted(heights, reverse=True), "the reveal only ever shrinks"
    assert 0 < heights[0] < full, "the first frame is partial, not a snap"
    assert heights[-1] == 0, "and it settles fully collapsed"
    assert reveal._t == 0.0


def test_collapsing_the_custom_row_cancels_a_live_edit_and_unfocuses_it(app, view):
    """Leaving Custom discards an uncommitted edit: the FIRST collapse tick calls
    the inner row's cancel_edit, dropping focus and restoring the stored value —
    without it a collapsed invisible field would keep eating keystrokes."""
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    row.input.focused = True
    row.input.text = "half-typed.example"
    app.settings._apply_server_mode("official")
    _settle(app, view)
    assert reveal._t == 0.0
    assert row.input.focused is False
    assert row.current_text() == env.get_custom_server_addr()


def test_the_collapse_cancels_the_edit_on_the_first_frame_not_at_the_settle(app, view):
    """Flip to Official and leave Options on the very next frame: the cancel used
    to wait for the ~17-frame settle, so commit_options_exit still read the
    'discarded' text out of the field and committed it. The first tick with
    visible_getter() False must cancel the edit, long before the settle."""
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    row.input.focused = True
    row.input.text = "discard.example:9"
    app.settings._apply_server_mode("official")
    view.draw(app.window, app.menu._menu_layout)
    assert 0.0 < reveal._t < 1.0, "one frame in, the collapse is nowhere near settled"
    assert row.input.focused is False
    assert row.current_text() == env.get_custom_server_addr()
    app.menu.goto_view("play")
    assert env.get_custom_server_addr() == "localhost:8000", \
        "the discarded text never reaches the exit commit"
    assert env.get_server_addr() == env.official_server_addr()


def test_a_click_elsewhere_blurs_the_open_custom_field(app, view):
    """A click that misses the reveal slot must still reach the inner row as a
    miss so its blur path runs — _live() used to short-circuit it, leaving the
    caret blinking in the custom field after clicking another row. The parked
    edit survives the blur; only focus goes."""
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    view.draw(app.window, app.menu._menu_layout)
    row.input.text = ""
    _type(row, "kept.example")
    assert row.input.focused is True
    elsewhere = (reveal._rect.centerx, reveal._rect.bottom + 60)
    assert not reveal._rect.collidepoint(elsewhere)
    assert not row._field_rect.collidepoint(elsewhere)
    assert reveal.handle_click(elsewhere) is False, \
        "the miss is delivered but never swallowed"
    assert row.input.focused is False
    assert row.current_text() == "kept.example"


def test_a_collapsing_field_eats_no_keys(app, view):
    """Between the mode flip and the settle the reveal is still _t > 0, and the
    old key gate (_t > 0.0) let the doomed field keep eating keystrokes it would
    discard. Key delivery follows visible_getter(), so the field goes deaf the
    instant the flip lands."""
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    row.input.focused = True
    row.input.text = "stuck"
    app.settings._apply_server_mode("official")
    assert reveal._t == 1.0, "no tick has run yet — only the gate can refuse"
    ev = pg.event.Event(pg.KEYDOWN, key=pg.K_z, mod=0, unicode="z")
    assert view.handle_key(ev) is False
    assert reveal.handle_key(ev) is False
    assert row.input.text == "stuck", "nothing was typed into the doomed field"
    view.draw(app.window, app.menu._menu_layout)
    assert 0.0 < reveal._t < 1.0
    row.input.focused = True
    assert view.handle_key(ev) is False, "mid-collapse the field still gets no keys"
    assert row.input.text == env.get_custom_server_addr()


def test_a_collapsed_custom_row_swallows_no_clicks(app, view):
    """After the collapse the inner TextRow still holds its last-drawn rects;
    the RevealRow gate (_t == 0) must eat every click before they can refocus
    the invisible field. The input is focused going INTO the collapse so a
    click leaking through the gate would really flip focus back on — clicking
    the live post-collapse field rect is the target that proves it."""
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    view.draw(app.window, app.menu._menu_layout)
    stale_slot = pg.Rect(reveal._rect)
    stale_field = pg.Rect(row._field_rect)
    assert stale_field.width > 0, "the field really was drawn while open"
    row.input.focused = True
    app.settings._apply_server_mode("official")
    _settle(app, view)
    assert reveal._t == 0.0
    assert row.input.focused is False, "the collapse dropped focus before any click"
    for target in (stale_slot.center, stale_field.center, stale_field.topleft,
                   row._field_rect.center):
        assert reveal.handle_click(target) is False
        assert reveal.contains_control(target) is False
        assert row.input.focused is False, "no click can refocus the invisible field"


def test_a_collapsed_custom_row_swallows_no_keys(app, view, monkeypatch):
    monkeypatch.setenv("CHESS_SERVER_MODE", "official")
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    row.input.focused = True
    row.input.text = "stuck"
    ev = pg.event.Event(pg.KEYDOWN, key=pg.K_z, mod=0, unicode="z")
    assert view.handle_key(ev) is False
    assert row.input.text == "stuck", "nothing was typed into the hidden field"
    assert reveal.handle_key(ev) is False
    assert reveal.cancel_edit() is False, \
        "a collapsed row declines Esc so it falls through to leaving the sub-view"


def test_a_mid_animation_click_below_the_clipped_slot_misses_the_field(app, view, monkeypatch):
    """Mid-reveal the inner row is drawn at full height and clipped to the
    shrunken slot, so its field rect sticks out below the visible sliver. The
    hit test follows the drawn slot, not the field, so that overhang is dead."""
    monkeypatch.setenv("CHESS_SERVER_MODE", "official")
    reveal = _online_rows(app, view)[1]
    row = app.settings._custom_server_row
    monkeypatch.setenv("CHESS_SERVER_MODE", "custom")
    view.draw(app.window, app.menu._menu_layout)
    slot = pg.Rect(reveal._rect)
    assert 0 < reveal._t < 0.5, "one frame in, the slot is still a sliver"
    below = (slot.centerx, slot.bottom + 3)
    assert row._field_rect.collidepoint(below), \
        "the clipped-away part of the field really covers that point"
    assert reveal.handle_click(below) is False
    assert reveal.contains_control(below) is False
    assert row.input.focused is False


def test_the_card_divider_is_skipped_above_a_fully_collapsed_row(app, monkeypatch):
    """A zero-height row draws no hairline: dividers only separate rows that
    are actually drawn, and a collapsed FIRST row leaves no line at the card
    top. Also pins the tick-before-heights ordering in _draw_card: the card
    outline recorded for a mid-animation frame must agree with the row heights
    measured right after that same frame's tick."""
    surface = pg.display.get_surface()
    state = {"open": False}
    inner = TextRow("Custom address", "desc", surface, lambda: "localhost:8000")
    reveal = RevealRow(inner, lambda: state["open"])
    seg = SegmentedRow("Server", "", [("A", "a"), ("B", "b")], lambda: "a", lambda k: None)
    toggle = ToggleRow("Hide", "desc", lambda: False, lambda v: None)
    rows = [reveal, seg, toggle]
    body = OptionsBody()
    body.set_sections([("Online", rows)])
    rect = pg.Rect(40, 40, 460, 400)
    content_x = rect.x + CARD_PAD_X
    content_w = rect.width - SCROLLBAR_RESERVE - 2 * CARD_PAD_X

    lines = []
    real_line = pg.draw.line
    monkeypatch.setattr(
        pg.draw, "line",
        lambda win, color, start, end, *a, **k: (
            lines.append((start, end)), real_line(win, color, start, end, *a, **k))[1])

    def dividers():
        return [s for s, e in lines
                if s[0] == content_x and e[0] == content_x + content_w]

    fonts = _fonts()
    body.draw(surface, rect, fonts)
    assert len(dividers()) == 1, "collapsed first row: one divider (seg|toggle), none on top"

    state["open"] = True
    lines.clear()
    body.draw(surface, rect, fonts)
    assert 0.0 < reveal._t < 1.0
    card = body._card_rects[-1]
    assert card.height == sum(r.height(fonts) for r in rows), \
        "tick ran before heights: outline and drawn geometry agree the same frame"

    for _ in range(60):
        body.draw(surface, rect, fonts)
    lines.clear()
    body.draw(surface, rect, fonts)
    assert len(dividers()) == 2, "fully open: three drawn rows, two dividers"


def test_body_scroll_stays_clamped_while_a_row_collapses(app):
    """Scrolled to the bottom, a collapsing row shrinks the content; the draw
    clamp reels the offset in every frame instead of leaving a void."""
    surface = pg.display.get_surface()
    state = {"open": True}
    inner = TextRow("Custom address", "desc", surface, lambda: "localhost:8000")
    reveal = RevealRow(inner, lambda: state["open"])
    fillers = [ToggleRow(f"opt {i}", "desc", lambda: False, lambda v: None)
               for i in range(12)]
    body = OptionsBody()
    body.set_sections([("Many", fillers + [reveal])])
    rect = pg.Rect(100, 60, 460, 240)
    fonts = _fonts()
    body.draw(surface, rect, fonts)
    body.handle_scroll((rect.centerx, rect.centery), -999)
    body.draw(surface, rect, fonts)
    at_bottom = body.scroll_offset
    assert at_bottom == body._max_scroll() > 0
    state["open"] = False
    for _ in range(60):
        body.draw(surface, rect, fonts)
        assert 0 <= body.scroll_offset <= at_bottom
    body.draw(surface, rect, fonts)
    assert reveal.height(fonts) == 0
    assert body.scroll_offset == body._max_scroll(), \
        "the offset re-clamps to the shrunken content"


def test_reveal_animation_frames_mint_no_new_card_surfaces(monkeypatch):
    """cut_rect_surface memoizes on exact pixel size, so a card whose height
    changed every reveal frame minted ~17 fresh 4x-supersampled full-card
    surfaces per toggle and the size-keyed cache retained them all. While any
    row is animating, the background renders at the card's full (all-open)
    height and blits clipped to the true card rect — so a whole toggle touches
    exactly the two settled sizes and the animation frames mint nothing new.
    Settled frames still request their exact size, pixel-identical to before."""
    surface = pg.display.get_surface()
    state = {"open": True}
    inner = TextRow("Custom address", "desc", surface, lambda: "localhost:8000")
    reveal = RevealRow(inner, lambda: state["open"])
    toggle = ToggleRow("Hide", "desc", lambda: False, lambda v: None)
    body = OptionsBody()
    body.set_sections([("Online", [reveal, toggle])])
    rect = pg.Rect(40, 40, 460, 400)
    fonts = _fonts()
    card_w = rect.width - SCROLLBAR_RESERVE
    card_sizes = []
    real = options_rows.cut_rect_surface

    def recording(size, *args, **kwargs):
        if size[0] == card_w:
            card_sizes.append(tuple(size))
        return real(size, *args, **kwargs)

    monkeypatch.setattr(options_rows, "cut_rect_surface", recording)
    body.draw(surface, rect, fonts)
    open_size = card_sizes[-1]
    assert open_size == (card_w, reveal.height(fonts) + toggle.height(fonts))
    state["open"] = False
    card_heights = set()
    for _ in range(60):
        body.draw(surface, rect, fonts)
        card_heights.add(body._card_rects[0].height)
    assert reveal._t == 0.0
    closed_size = card_sizes[-1]
    assert closed_size == (card_w, toggle.height(fonts))
    assert len(card_heights) > 4, "the collapse really animated through many heights"
    assert set(card_sizes) == {open_size, closed_size}, \
        "animation frames reuse the settled sizes instead of minting per-frame ones"


def test_a_relayout_mid_animation_keeps_the_reveal_progress(app, view, monkeypatch):
    monkeypatch.setenv("CHESS_SERVER_MODE", "official")
    reveal = _online_rows(app, view)[1]
    monkeypatch.setenv("CHESS_SERVER_MODE", "custom")
    view.draw(app.window, app.menu._menu_layout)
    view.draw(app.window, app.menu._menu_layout)
    mid = reveal._t
    assert 0.0 < mid < 1.0
    view.relayout(app.menu._menu_layout)
    assert reveal._t == mid, "relayout clears fonts, not animation progress"
    view.draw(app.window, app.menu._menu_layout)
    assert reveal._t > mid, "and the reveal keeps opening from where it was"


# --- #89 session lock: no server switch while connected or queued -----------

def test_switching_server_mode_during_an_online_session_toasts_and_keeps_the_mode(
        app, view, monkeypatch):
    seg = _online_rows(app, view)[0]
    _draw_row(seg, pg.Rect(40, 40, 520, 60))
    monkeypatch.setattr(app.coordinator, "is_connected", lambda: True)
    assert seg.handle_click(seg._rects["official"].center) is True
    assert app.toast.message == SERVER_SWITCH_BLOCKED_MESSAGE
    assert env.get_server_mode() == "custom"
    assert "CHESS_SERVER_MODE" not in os.environ


def test_committing_a_custom_address_during_an_online_session_toasts_and_does_not_persist(
        app, view, monkeypatch):
    """The reconnect probe re-reads the resolved address on every attempt, so a
    mid-session rewrite would point /reclaim at the wrong host. Enter-commit in
    the field refuses and toasts; the typed text stays in the field (edge 6)."""
    _online_rows(app, view)
    row = app.settings._custom_server_row
    monkeypatch.setattr(app.coordinator, "is_connected", lambda: True)
    row.input.focused = True
    row.input.text = "evil.example:9"
    row.handle_key(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, mod=0, unicode="\r"))
    assert app.toast.message == SERVER_SWITCH_BLOCKED_MESSAGE
    assert env.get_custom_server_addr() == "localhost:8000"
    assert "CHESS_CUSTOM_SERVER_ADDR" not in os.environ
    assert row.current_text() == "evil.example:9"


def test_committing_a_custom_address_in_official_mode_persists_quietly_without_a_toast(
        app, view, monkeypatch):
    """The exit commit still saves a custom address typed while Official is
    active — it just never announces a server the player is not pointed at."""
    env.set_server_mode("official")
    _online_rows(app, view)
    app.settings._custom_server_row.input.text = "my.box:9000"
    app.menu.goto_view("play")
    assert env.get_custom_server_addr() == "my.box:9000"
    assert env.get_server_addr() == env.official_server_addr()
    assert app.toast.is_visible() is False


# --- #90 test-connection probe ----------------------------------------------

def _monotonic_seq(values):
    remaining = list(values)

    def fn():
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return fn


def _probe_harness(monkeypatch, *, health, monotonic=(10.0, 10.25), inline=True):
    """Patch the settings module's thread/probe/clock seams. Replaces the
    MODULE BINDINGS (settings_module.threading / .time), never the global
    stdlib modules, so nothing else in the process sees the fakes. Returns
    (created_threads, probe_calls)."""
    created = []
    calls = []

    class _FakeThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            created.append(self)

        def start(self):
            if inline:
                self.run_now()

        def run_now(self):
            self.target(*self.args, **self.kwargs)

    def fake_probe(addr, timeout=HEALTHZ_TIMEOUT_SECONDS):
        calls.append((addr, timeout))
        return health

    monkeypatch.setattr(settings_module, "threading",
                        SimpleNamespace(Thread=_FakeThread, Lock=threading.Lock))
    monkeypatch.setattr(settings_module, "time",
                        SimpleNamespace(monotonic=_monotonic_seq(monotonic)))
    monkeypatch.setattr(settings_module, "probe_server_health", fake_probe)
    return created, calls


def test_test_connection_row_reports_ok_with_latency_and_app_version(app, monkeypatch):
    _probe_harness(monkeypatch, health={
        "status": "ok", "version": PROTOCOL_VERSION, "app_version": "9.9.9"})
    app.settings._on_test_server()
    assert app.settings._probe_status() == ("ok", "OK · 250 ms · v9.9.9")
    assert app.settings._probe_button_label() == SERVER_PROBE_BUTTON_IDLE


def test_test_connection_row_reports_a_full_server(app, monkeypatch):
    """A server at its room cap answers 200 with status=full, so without this
    branch the picker would call it OK and the player would only learn the
    truth from a 503 on Play."""
    _probe_harness(monkeypatch, health={
        "status": "full", "version": PROTOCOL_VERSION, "app_version": "9.9.9"})
    app.settings._on_test_server()
    assert app.settings._probe_status() == ("warn", "Server full · 250 ms")


def test_test_connection_row_reports_a_degraded_server(app, monkeypatch):
    """Degraded means the server is answering but has fallen behind on the work
    that ends games on time -- playable, but not something to pick over a
    healthy one, so it reads as a warning rather than an OK."""
    _probe_harness(monkeypatch, health={
        "status": "degraded", "version": PROTOCOL_VERSION, "app_version": "9.9.9"})
    app.settings._on_test_server()
    assert app.settings._probe_status() == ("warn", "Degraded · 250 ms")


def test_test_connection_row_reports_unreachable(app, monkeypatch):
    _probe_harness(monkeypatch, health=None)
    app.settings._on_test_server()
    assert app.settings._probe_status() == ("warn", "Unreachable")


def test_test_connection_row_reports_a_protocol_mismatch_naming_both_versions(app, monkeypatch):
    _probe_harness(monkeypatch, health={
        "status": "ok", "version": PROTOCOL_VERSION - 1, "app_version": "1.0"})
    app.settings._on_test_server()
    tone, text = app.settings._probe_status()
    assert tone == "warn"
    assert text == "Protocol mismatch: server v{} ≠ client v{}".format(
        PROTOCOL_VERSION - 1, PROTOCOL_VERSION)


def test_test_connection_button_shows_a_pending_label_while_a_probe_is_in_flight(
        app, monkeypatch):
    created, _calls = _probe_harness(
        monkeypatch, health={"status": "ok", "version": PROTOCOL_VERSION,
                             "app_version": ""}, inline=False)
    app.settings._on_test_server()
    assert app.settings._probe_button_label() == SERVER_PROBE_BUTTON_PENDING
    assert app.settings._probe_status() == ("idle", ""), "no stale result while pending"
    created[0].run_now()
    assert app.settings._probe_button_label() == SERVER_PROBE_BUTTON_IDLE
    assert app.settings._probe_status() == ("ok", "OK · 250 ms")


def test_test_connection_is_debounced_while_a_probe_is_in_flight(app, monkeypatch):
    created, _calls = _probe_harness(monkeypatch, health=None, inline=False)
    app.settings._on_test_server()
    app.settings._on_test_server()
    assert len(created) == 1, "a double click cannot spawn two probe threads"
    created[0].run_now()
    app.settings._on_test_server()
    assert len(created) == 2, "once resolved, the button works again"


def test_a_stale_probe_generation_is_discarded(app, monkeypatch):
    created, _calls = _probe_harness(
        monkeypatch, health={"status": "ok", "version": PROTOCOL_VERSION,
                             "app_version": ""}, inline=False)
    app.settings._on_test_server()
    app.settings._reset_server_probe()
    created[0].run_now()
    assert app.settings._probe_status() == ("idle", ""), \
        "a worker outrun by a reset never lands its result"
    assert app.settings._probe_button_label() == SERVER_PROBE_BUTTON_IDLE


def test_the_probe_targets_the_typed_custom_address_not_the_stored_one(
        app, view, monkeypatch):
    _online_rows(app, view)
    app.settings._custom_server_row.input.text = "typed.example:9  "
    _created, calls = _probe_harness(monkeypatch, health=None)
    app.settings._on_test_server()
    assert calls == [("typed.example:9", HEALTHZ_TIMEOUT_SECONDS)]
    assert env.get_custom_server_addr() == "localhost:8000", \
        "probing never commits the typed text"


def test_the_probe_targets_the_official_address_in_official_mode(app, view, monkeypatch):
    """Official mode probes env.get_server_addr() — the resolved active key,
    exactly what transport will dial — not the official constant. Switching
    modes through the controller resolves that key, so with no override it
    still lands on the official address; the typed custom text is ignored."""
    app.settings._apply_server_mode("official")
    _online_rows(app, view)
    app.settings._custom_server_row.input.text = "typed.example:9"
    _created, calls = _probe_harness(monkeypatch, health=None)
    app.settings._on_test_server()
    assert env.get_server_addr() == env.official_server_addr()
    assert calls == [(env.get_server_addr(), HEALTHZ_TIMEOUT_SECONDS)]


def test_a_launch_time_addr_override_redirects_the_official_mode_probe(
        app, view, monkeypatch):
    """A CHESS_SERVER_ADDR set at launch wins over the resolved official
    address, and the probe must test the host the client would actually dial —
    probing the constant would report OK for a server the player never
    reaches."""
    app.settings._apply_server_mode("official")
    monkeypatch.setenv("CHESS_SERVER_ADDR", "override.example:9001")
    _online_rows(app, view)
    _created, calls = _probe_harness(monkeypatch, health=None)
    app.settings._on_test_server()
    assert calls == [("override.example:9001", HEALTHZ_TIMEOUT_SECONDS)]


def test_the_probe_result_is_cleared_on_options_exit(app, view, monkeypatch):
    _online_rows(app, view)
    _probe_harness(monkeypatch, health={
        "status": "ok", "version": PROTOCOL_VERSION, "app_version": ""})
    app.settings._on_test_server()
    assert app.settings._probe_status()[0] == "ok"
    app.menu.goto_view("play")
    assert app.settings._probe_status() == ("idle", "")


def test_typing_in_the_custom_field_blanks_a_verdict_probed_for_another_address(
        app, view, monkeypatch):
    """The stored probe result is keyed to the address it probed: the moment the
    field's current target drifts away from that key the verdict blanks —
    otherwise an OK earned by localhost keeps glowing next to a freshly typed
    stranger. Retyping the probed address matches the key again and the verdict
    comes straight back."""
    _online_rows(app, view)
    row = app.settings._custom_server_row
    probed = row.current_text()
    _probe_harness(monkeypatch, health={
        "status": "ok", "version": PROTOCOL_VERSION, "app_version": ""})
    app.settings._on_test_server()
    assert app.settings._probe_status()[0] == "ok"
    row.input.text = ""
    _type(row, "other.example:9")
    assert app.settings._probe_status() == ("idle", ""), \
        "the verdict was earned by another address, so it blanks"
    row.input.text = ""
    _type(row, probed)
    assert app.settings._probe_status()[0] == "ok", \
        "retyping the probed address shows the stored verdict again"


def test_the_probe_result_is_cleared_when_the_server_target_changes(app, view, monkeypatch):
    _online_rows(app, view)
    _probe_harness(monkeypatch, health={
        "status": "ok", "version": PROTOCOL_VERSION, "app_version": ""})
    app.settings._on_test_server()
    assert app.settings._probe_status()[0] == "ok"
    app.settings._apply_server_mode("official")
    assert app.settings._probe_status() == ("idle", ""), "mode switch clears the verdict"
    app.settings._on_test_server()
    assert app.settings._probe_status()[0] == "ok"
    app.settings._commit_custom_server_addr("other.example:9")
    assert app.settings._probe_status() == ("idle", ""), "address commit clears it too"


def test_the_probe_runs_on_a_daemon_thread_with_the_resolved_address(app, view, monkeypatch):
    """The worker must never be able to keep the app alive, and it captures the
    target address on the UI thread before the thread starts."""
    _online_rows(app, view)
    app.settings._custom_server_row.input.text = "typed.example:9"
    created, calls = _probe_harness(monkeypatch, health=None, inline=False)
    app.settings._on_test_server()
    assert created[0].daemon is True
    assert created[0].args[0] == "typed.example:9"
    created[0].run_now()
    assert calls == [("typed.example:9", HEALTHZ_TIMEOUT_SECONDS)]
