"""ConfirmModal on the shared shell: danger intent rail, ghost(cancel)/primary
(affirmative) button order, a wrapped sub line, and click routing that hides
before firing the callback."""

import pygame as pg

from tests.conftest import pygame_display
from chessshootout.frontend.modals.confirm import ConfirmModal
from chessshootout.frontend.visual.colors import Colors


_pygame_init = pygame_display(900, 700)


def _has_color(win, rect, hexcolor, tol=30):
    want = pg.Color(hexcolor)
    rect = rect.clip(win.get_rect())
    for x in range(rect.x, rect.right, 2):
        for y in range(rect.y, rect.bottom, 2):
            c = win.get_at((x, y))
            if (abs(c.r - want.r) <= tol and abs(c.g - want.g) <= tol
                    and abs(c.b - want.b) <= tol):
                return True
    return False


def _modal():
    m = ConfirmModal(pg.display.get_surface())
    m.set_rect(pg.Rect(230, 200, 440, 260))
    return m


def test_not_visible_until_shown():
    m = _modal()
    assert m.is_visible() is False
    m.show("Tap out?", lambda: None)
    assert m.is_visible() is True


def test_danger_shows_red_rail():
    m = _modal()
    m.show("Tap out?", lambda: None, danger=True)
    m.window.fill((0, 0, 0))
    m.draw()
    rail = pg.Rect(m._panel.x + 24, m._panel.y + 1, 120, 4)
    assert _has_color(m.window, rail, Colors.loss, tol=45)


def test_non_danger_rail_is_accent():
    m = _modal()
    m.show("Offer a draw?", lambda: None)
    m.window.fill((0, 0, 0))
    m.draw()
    rail = pg.Rect(m._panel.x + 24, m._panel.y + 1, 120, 4)
    assert _has_color(m.window, rail, Colors.accent, tol=45)


def test_panel_is_compact_relative_to_rect():
    """The panel self-sizes to its content and centres in the allotted rect,
    rather than filling a large modal rect."""
    m = ConfirmModal(pg.display.get_surface())
    m.set_rect(pg.Rect(230, 120, 440, 460))
    m.show("Tap out?", lambda: None, sub="The pieces are watching.", danger=True)
    m.draw()
    assert m._panel.height < m.rect.height - 60
    assert abs(m._panel.centery - m.rect.centery) <= 1


def test_affirmative_is_primary_and_on_the_right():
    m = _modal()
    m.show("Tap out?", lambda: None, yes_label="I'm done", no_label="Keep fighting")
    m.window.fill((0, 0, 0))
    m.draw()
    assert _has_color(m.window, m.button_rects["yes"].inflate(-8, -8), Colors.accent, 20)
    assert m.button_rects["no"].x < m.button_rects["yes"].x


def test_an_empty_cancel_label_leaves_that_button_out():
    """The one-answer card (Update required): nothing about it can be cancelled,
    so a second button would only be a second way to say OK. An empty cancel
    label drops the button from the row the same way on_extra=None drops the
    third one -- and out of the hit map with it, since the rects the row hands
    back ARE what clicks are tested against."""
    fired = []
    m = _modal()
    m.show("Update required", lambda: fired.append("yes"),
           on_no=lambda: fired.append("no"), yes_label="OK", no_label="")
    m.window.fill((0, 0, 0))
    m.draw()

    assert set(m.button_rects) == {"yes"}
    assert abs(m.button_rects["yes"].centerx - m._panel.centerx) <= 1, "centred"
    assert _has_color(m.window, m.button_rects["yes"].inflate(-8, -8), Colors.accent, 20)

    m.handle_click(m.button_rects["yes"].center)

    assert fired == ["yes"]
    assert m.is_visible() is False


def test_sub_body_renders():
    m = _modal()
    m.window.fill((0, 0, 0))
    m.show("Tap out?", lambda: None)
    m.draw()
    bare = pg.image.tobytes(m.window.subsurface(m.rect), "RGB")
    m.window.fill((0, 0, 0))
    m.show("Tap out?", lambda: None, sub="Throw in the towel and hand them the W?")
    m.draw()
    assert pg.image.tobytes(m.window.subsurface(m.rect), "RGB") != bare


def test_click_hides_then_fires_callback():
    fired = []
    m = _modal()
    m.show("Tap out?", on_yes=lambda: fired.append("yes"),
           on_no=lambda: fired.append("no"))
    m.draw()
    assert m.handle_click(m.button_rects["yes"].center) is True
    assert fired == ["yes"]
    assert m.is_visible() is False


def test_extra_button_appended_last():
    m = _modal()
    m.show("Move games?", lambda: None, on_no=lambda: None,
           on_extra=lambda: None, yes_label="Move", no_label="Don't move",
           extra_label="Cancel")
    m.draw()
    assert set(m.button_rects) == {"no", "yes", "extra"}
    assert m.button_rects["extra"].x > m.button_rects["yes"].x


def test_emoji_icon_grows_panel_and_renders_tile():
    plain = _modal()
    plain.show("Tap out?", lambda: None)
    plain.draw()
    plain_h = plain._panel.height

    iconed = _modal()
    iconed.show("Tap out?", lambda: None, emoji="🏳️")
    iconed.window.fill((0, 0, 0))
    iconed.draw()
    assert iconed._panel.height > plain_h
    tile_band = pg.Rect(iconed._panel.centerx - 30, iconed._panel.y + 8, 60, 60)
    assert _has_color(iconed.window, tile_band, Colors.surface_hover, tol=18)
