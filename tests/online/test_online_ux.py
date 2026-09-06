"""M16: Online UX layer.

Animation duration scales with the time control, the wait modal shows
elapsed time during search, a themed match-found VS reveal holds for a few
seconds before the game starts, the reconnecting on-board overlay surfaces
only while the client is reconnecting mid-game, and transient errors show a
toast (not a modal).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pygame as pg
import pytest

from tests.conftest import pygame_display
from tests.helpers import online_start_payload
from chessshootout.backend.utils import BOARD_SIZE
from chessshootout.frontend.frontend import Frontend
from chessshootout.frontend import online_coordinator as coordinator_module
from chessshootout.frontend.online_coordinator import RECONNECT_MODAL_DEBOUNCE_MS
from chessshootout.frontend.screens.game import (
    ANIM_MS_DEFAULT, ANIM_MS_MIN, ANIM_MS_MAX, compute_animation_ms,
)
from chessshootout.frontend.visual.colors import Colors
from chessshootout.frontend.modals.reconnecting import ReconnectingModal
from chessshootout.frontend.online_coordinator import (
    MATCH_FOUND_SECONDS, NOT_YOUR_TURN_TOASTS, ONLINE_GAME_STATE_REASONS,
    ONLINE_HARD_FAILURE_LABELS, ONLINE_HARD_FAILURE_REASONS,
    ONLINE_TRANSIENT_REASON_LABELS, UPDATE_REQUIRED_PROTOCOL_SUB,
    UPDATE_REQUIRED_TITLE, UPDATE_REQUIRED_UNKNOWN_SUB,
)
from chessshootout.server.protocol import PROTOCOL_VERSION, Reason


_pygame_init = pygame_display(600, 400)


@pytest.fixture
def frontend():
    fe = Frontend(900, 600)
    yield fe
    pg.display.set_mode((600, 400))


@pytest.mark.parametrize(
    "seconds, expected",
    [
        pytest.param(None, ANIM_MS_DEFAULT, id="no_clock_uses_default"),
        pytest.param(0, ANIM_MS_DEFAULT, id="zero_clock_uses_default"),
        pytest.param(60, ANIM_MS_MIN, id="bullet_1plus0_clamps_to_min"),
        pytest.param(300, 150, id="blitz_5plus0_in_range_as_is"),
        pytest.param(600, ANIM_MS_MAX, id="rapid_10plus0_clamps_to_max"),
        pytest.param(3600, ANIM_MS_MAX, id="classical_still_capped_at_max"),
    ],
)
def test_compute_animation_ms(seconds, expected):
    assert compute_animation_ms(seconds) == expected


@pytest.mark.parametrize(
    "time_control, expected",
    [
        pytest.param((300, 0), compute_animation_ms(300), id="time_control_drives_anim_ms"),
        pytest.param(None, ANIM_MS_DEFAULT, id="no_clock_uses_default"),
    ],
)
def test_reset_to_new_game_sets_anim_ms(frontend, time_control, expected):
    frontend.game._time_control = time_control
    frontend.game._reset_to_new_game()
    assert frontend.game.board.animation_duration_ms == expected


def test_reconnecting_modal_starts_hidden():
    m = ReconnectingModal(pg.display.get_surface())
    assert not m.is_visible()


def test_reconnecting_modal_show_makes_visible_and_caches_callback():
    m = ReconnectingModal(pg.display.get_surface())
    m.set_rect(pg.Rect(0, 0, 400, 400))
    cancelled = []
    m.show(pg.time.get_ticks(), on_abandon=lambda: cancelled.append(True))
    assert m.is_visible()
    m.draw()
    m.handle_click(m.button_rects["abandon"].center)
    assert cancelled == [True]


def test_reconnecting_overlay_fills_board_rect_with_scrim():
    """The overlay darkens the whole board rect (scrim) and paints an amber
    spinner ring — both absent before show()."""
    win = pg.display.get_surface()
    m = ReconnectingModal(win)
    rect = pg.Rect(0, 0, 420, 360)
    m.set_rect(rect)
    m.show(pg.time.get_ticks(), on_abandon=lambda: None)

    win.fill((0, 0, 0))
    m.draw()
    scrim_corner = win.get_at((4, 4))[:3]
    assert scrim_corner != (0, 0, 0)

    amber = pg.Color(Colors.amber)[:3]

    def amber_pixels():
        return sum(
            1
            for x in range(0, rect.width, 2)
            for y in range(0, rect.height, 2)
            if max(abs(win.get_at((x, y))[i] - amber[i]) for i in range(3)) < 24
        )

    assert amber_pixels() > 0


def _reconnecting_client():
    ns = SimpleNamespace(
        state="reconnecting",
        is_server_silent=lambda: False,
        heartbeat_interval=lambda: 2.0,
        send_ping=lambda ply: None,
    )
    ns.is_connected = lambda: ns.state == "connected"
    return ns


def test_reconnecting_overlay_appears_after_debounce(frontend, monkeypatch):
    fake_now = [10_000]
    monkeypatch.setattr(pg.time, "get_ticks", lambda: fake_now[0])
    frontend.game.variant = "online"
    frontend.coordinator.client = _reconnecting_client()
    frontend.coordinator._update_online_phase()                       # arms the debounce clock
    assert not frontend.coordinator.reconnecting_modal.is_visible(), \
        "debounced, not shown instantly"
    fake_now[0] += RECONNECT_MODAL_DEBOUNCE_MS + 50
    frontend.coordinator._update_online_phase()
    assert frontend.coordinator.reconnecting_modal.is_visible()


def test_brief_blip_never_flashes_the_modal(frontend, monkeypatch):
    fake_now = [10_000]
    monkeypatch.setattr(pg.time, "get_ticks", lambda: fake_now[0])
    frontend.game.variant = "online"
    client = _reconnecting_client()
    frontend.coordinator.client = client
    frontend.coordinator._update_online_phase()
    fake_now[0] += RECONNECT_MODAL_DEBOUNCE_MS - 100       # recover before the threshold
    client.state = "connected"
    frontend.coordinator._update_online_phase()
    assert not frontend.coordinator.reconnecting_modal.is_visible()


def test_reconnecting_overlay_hides_immediately_on_recovery(frontend, monkeypatch):
    fake_now = [10_000]
    monkeypatch.setattr(pg.time, "get_ticks", lambda: fake_now[0])
    frontend.game.variant = "online"
    client = _reconnecting_client()
    frontend.coordinator.client = client
    frontend.coordinator._update_online_phase()
    fake_now[0] += RECONNECT_MODAL_DEBOUNCE_MS + 50
    frontend.coordinator._update_online_phase()
    assert frontend.coordinator.reconnecting_modal.is_visible()
    client.state = "connected"
    frontend.coordinator._update_online_phase()
    assert not frontend.coordinator.reconnecting_modal.is_visible(), \
        "hidden the moment the socket is back — no input-blocking linger"


def test_reconnecting_overlay_cancel_calls_abandon(frontend, monkeypatch):
    fake_now = [10_000]
    monkeypatch.setattr(pg.time, "get_ticks", lambda: fake_now[0])
    abandoned = []
    monkeypatch.setattr(frontend.coordinator, "_abandon_online_game",
                        lambda: abandoned.append(True))
    frontend.game.variant = "online"
    frontend.coordinator.client = _reconnecting_client()
    frontend.coordinator._update_online_phase()
    fake_now[0] += RECONNECT_MODAL_DEBOUNCE_MS + 50
    frontend.coordinator._update_online_phase()
    frontend.coordinator.reconnecting_modal.draw()
    frontend.coordinator.reconnecting_modal.handle_click(
        frontend.coordinator.reconnecting_modal.button_rects["abandon"].center,
    )
    assert abandoned == [True]


def test_wait_modal_elapsed_tracks_search_time(frontend, monkeypatch):
    fake_now = [10000]
    monkeypatch.setattr(pg.time, "get_ticks", lambda: fake_now[0])
    frontend.coordinator.wait_modal.show("Rapid", "10 + 5", on_cancel=lambda: None)
    frontend.coordinator._wait_started_at_ms = fake_now[0]
    fake_now[0] += 7 * 1000
    frontend.coordinator._update_online_phase()
    assert frontend.coordinator.wait_modal.elapsed == 7
    fake_now[0] += 60 * 1000
    frontend.coordinator._update_online_phase()
    assert frontend.coordinator.wait_modal.elapsed == 67


def test_match_found_reveal_holds_then_starts_game(frontend, monkeypatch):
    fake_now = [50000]
    monkeypatch.setattr(pg.time, "get_ticks", lambda: fake_now[0])
    started = []
    monkeypatch.setattr(frontend.coordinator, "_start_online_game",
                        lambda payload: started.append(payload))
    frontend.coordinator.wait_modal.show("Rapid", "10 + 5", on_cancel=lambda: None)
    frontend.coordinator._wait_started_at_ms = fake_now[0]

    payload = online_start_payload(white_name="A", black_name="B")
    frontend.coordinator._begin_match_found_transition(payload)

    assert frontend.coordinator.match_found_modal.is_visible()
    assert not frontend.coordinator.wait_modal.is_visible()
    fake_now[0] += MATCH_FOUND_SECONDS * 1000 - 50
    frontend.coordinator._update_online_phase()
    assert started == []
    assert frontend.coordinator.match_found_modal.is_visible()

    fake_now[0] += 100
    frontend.coordinator._update_online_phase()
    assert started == [payload]
    assert not frontend.coordinator.match_found_modal.is_visible()
    fake_now[0] += 1000
    frontend.coordinator._update_online_phase()
    assert started == [payload]


def test_match_found_transition_plays_online_game_start_sound(frontend):
    plays = []
    frontend.sound_manager.play_online_game_start = lambda: plays.append(True)
    frontend.coordinator.wait_modal.show("Rapid", "10 + 5", on_cancel=lambda: None)
    frontend.coordinator._wait_started_at_ms = pg.time.get_ticks()
    frontend.coordinator._begin_match_found_transition(
        online_start_payload(white_name="A", black_name="B"))
    assert plays == [True]


@pytest.mark.parametrize(
    "reason, expected_title",
    [
        pytest.param("server_unreachable",
                     ONLINE_HARD_FAILURE_LABELS["server_unreachable"],
                     id="server_unreachable_friendly_label"),
        pytest.param("reconnect_failed",
                     ONLINE_HARD_FAILURE_LABELS["reconnect_failed"],
                     id="reconnect_failed_friendly_label"),
        pytest.param(Reason.INVALID_FIELD,
                     ONLINE_HARD_FAILURE_LABELS[Reason.INVALID_FIELD],
                     id="invalid_field_friendly_label"),
        pytest.param("http_503", "Server unreachable",
                     id="http_prefixed_falls_back_to_generic"),
    ],
)
def test_hard_failure_shows_confirm_modal_with_friendly_label(
    frontend, reason, expected_title,
):
    """Hard failures surface a confirm modal with readable text (no raw engine
    code in the title) and never eat the event into a toast."""
    frontend.coordinator._handle_online_error({"reason": reason})
    assert frontend.confirm_modal.is_visible()
    assert frontend.confirm_modal.title == expected_title
    assert reason not in frontend.confirm_modal.title
    assert frontend.toast.is_visible() is False


def test_an_outdated_build_shows_the_update_card_naming_both_versions(
    frontend, monkeypatch,
):
    """The client_outdated wording is built from two PARSED versions, so the
    card can only ever print numbers this build understood."""
    monkeypatch.setattr(coordinator_module.paths, "get_app_version", lambda: "2.12.2")
    frontend.coordinator.wait_modal.show("Blitz", "5 + 0", on_cancel=lambda: None)

    frontend.coordinator._handle_online_error(
        {"reason": Reason.CLIENT_OUTDATED, "min_version": "2.13.0"})

    assert frontend.confirm_modal.is_visible()
    assert frontend.confirm_modal.title == UPDATE_REQUIRED_TITLE
    assert frontend.confirm_modal.sub == (
        "You run v2.12.2 · this server needs v2.13.0 or newer — update your install")
    assert frontend.confirm_modal.yes_label == "OK"
    assert frontend.confirm_modal.emoji is None
    assert not frontend.coordinator.wait_modal.is_visible(), \
        "the search card comes down, or the player watches a dead spinner"
    assert not frontend.toast.is_visible()


@pytest.mark.parametrize(
    "app_version, payload",
    [
        pytest.param("2.12.2", {"reason": Reason.CLIENT_OUTDATED,
                                "min_version": "9.9.9\nUpdate at evil.example"},
                     id="a_forged_minimum_is_never_rendered"),
        pytest.param("2.12.2", {"reason": Reason.CLIENT_OUTDATED,
                                "min_version": "‮drop everything"},
                     id="control_characters_are_never_rendered"),
        pytest.param("2.12.2", {"reason": Reason.CLIENT_OUTDATED, "min_version": 213},
                     id="a_minimum_that_is_not_text_is_never_rendered"),
        pytest.param("2.12.2", {"reason": Reason.CLIENT_OUTDATED},
                     id="no_minimum_named_at_all"),
        pytest.param("", {"reason": Reason.CLIENT_OUTDATED, "min_version": "2.13.0"},
                     id="a_source_run_has_no_version_of_its_own_to_name"),
    ],
)
def test_the_update_card_falls_back_rather_than_print_an_unparsed_version(
    frontend, monkeypatch, app_version, payload,
):
    """The sub-line is server-influenced text drawn full width on the player's
    screen. Anything that does not parse as a version drops the whole sentence
    for the generic one instead of being echoed."""
    monkeypatch.setattr(coordinator_module.paths, "get_app_version", lambda: app_version)

    frontend.coordinator._handle_online_error(payload)

    assert frontend.confirm_modal.title == UPDATE_REQUIRED_TITLE
    assert frontend.confirm_modal.sub == UPDATE_REQUIRED_UNKNOWN_SUB


def test_a_protocol_gap_shows_the_update_card_with_direction_neutral_copy(frontend):
    """version_mismatch used to be swallowed by ONLINE_GAME_STATE_REASONS: the
    search card stayed up and spun for ever. It is an update card now, worded
    without blaming either side, since which of the two is older cannot be told
    from a refusal that names no server version."""
    frontend.coordinator.wait_modal.show("Blitz", "5 + 0", on_cancel=lambda: None)

    frontend.coordinator._handle_online_error({"reason": Reason.VERSION_MISMATCH})

    assert frontend.confirm_modal.is_visible()
    assert frontend.confirm_modal.title == UPDATE_REQUIRED_TITLE
    assert frontend.confirm_modal.sub == UPDATE_REQUIRED_PROTOCOL_SUB
    assert str(PROTOCOL_VERSION) in frontend.confirm_modal.sub
    assert not frontend.coordinator.wait_modal.is_visible()
    assert Reason.VERSION_MISMATCH not in ONLINE_GAME_STATE_REASONS


@pytest.mark.parametrize(
    "button",
    [
        pytest.param("yes", id="the_ok_button"),
        pytest.param("no", id="the_second_button_of_the_shared_shell"),
    ],
)
def test_dismissing_the_update_card_puts_the_player_back_on_the_play_card(
    frontend, monkeypatch, button,
):
    """The confirm shell always draws two buttons, so BOTH are wired to the same
    cancel: whichever one is clicked, the search is given up and the menu's play
    card comes back. A button that only hid the box would strand the player on a
    menu with no play card on it."""
    cancelled = []
    monkeypatch.setattr(frontend.coordinator, "_on_online_cancel",
                        lambda: cancelled.append(True))

    frontend.coordinator._handle_online_error({"reason": Reason.CLIENT_OUTDATED})
    frontend.confirm_modal.draw()
    frontend.confirm_modal.handle_click(
        frontend.confirm_modal.button_rects[button].center)

    assert cancelled == [True]
    assert not frontend.confirm_modal.is_visible()


def test_the_update_card_is_shown_once_per_refusal(frontend, caplog):
    """One refusal, one card and one WARNING -- the client stops after the
    refusal, so a repeat here would mean something re-entered the branch."""
    with caplog.at_level(logging.WARNING, logger="chess.frontend"):
        frontend.coordinator._handle_online_error({"reason": Reason.CLIENT_OUTDATED})
    warnings = [rec.getMessage() for rec in caplog.records
                if rec.getMessage().startswith("online update required")]
    assert warnings == [f"online update required reason={Reason.CLIENT_OUTDATED}"]
    assert frontend.confirm_modal.is_visible()


def test_room_lost_shows_new_search_modal(frontend, monkeypatch):
    restart_calls = []
    monkeypatch.setattr(frontend.coordinator, "_restart_online_search",
                        lambda: restart_calls.append(True))
    frontend.coordinator._handle_online_error({"reason": "room_lost"})
    assert frontend.confirm_modal.is_visible()
    assert "Server restarted" in frontend.confirm_modal.title
    assert frontend.confirm_modal.yes_label == "New Search"
    frontend.confirm_modal.draw()
    frontend.confirm_modal.handle_click(
        frontend.confirm_modal.button_rects["yes"].center,
    )
    assert restart_calls == [True]


def test_room_lost_cancel_returns_to_menu(frontend, monkeypatch):
    abandoned = []
    monkeypatch.setattr(frontend.coordinator, "_abandon_online_game",
                        lambda: abandoned.append(True))
    frontend.coordinator._handle_online_error({"reason": "room_lost"})
    frontend.confirm_modal.draw()
    frontend.confirm_modal.handle_click(
        frontend.confirm_modal.button_rects["no"].center,
    )
    assert abandoned == [True]


@pytest.mark.parametrize(
    "reason, expected_message",
    [
        pytest.param("rate_limited",
                     ONLINE_TRANSIENT_REASON_LABELS["rate_limited"],
                     id="known_reason_maps_to_friendly_label"),
        pytest.param("already_in_game",
                     ONLINE_TRANSIENT_REASON_LABELS["already_in_game"],
                     id="already_in_game_maps_to_friendly_label"),
        pytest.param("rematch_already_pending",
                     ONLINE_TRANSIENT_REASON_LABELS["rematch_already_pending"],
                     id="rematch_already_pending_maps_to_friendly_label"),
        pytest.param("no_takeback_available",
                     ONLINE_TRANSIENT_REASON_LABELS["no_takeback_available"],
                     id="no_takeback_available_maps_to_friendly_label"),
        pytest.param("weird_thing", "weird_thing",
                     id="unknown_reason_falls_through_to_raw"),
    ],
)
def test_transient_error_shows_toast_not_modal(frontend, reason, expected_message):
    frontend.coordinator._handle_online_error({"reason": reason})
    assert frontend.toast.is_visible()
    assert frontend.toast.message == expected_message
    assert not frontend.confirm_modal.is_visible()


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"reason": "not_your_turn"}, id="bare_not_your_turn"),
        pytest.param({"reason": "not_your_turn", "msg_type": "weird_action"},
                     id="not_your_turn_unknown_msg_type"),
    ],
)
def test_not_your_turn_without_known_msg_type_stays_silent(frontend, payload):
    """A not_your_turn reply with no recognised msg_type is purely defensive
    (the client already gates by turn) — it must show neither modal nor toast."""
    frontend.coordinator._handle_online_error(payload)
    assert not frontend.confirm_modal.is_visible()
    assert not frontend.toast.is_visible()


@pytest.mark.parametrize(
    "msg_type",
    [
        pytest.param("takeback_request", id="takeback_explains_after_move_only"),
    ],
)
def test_not_your_turn_with_known_msg_type_shows_toast(frontend, msg_type):
    """When the rejected action is tagged with a msg_type the client knows, a
    friendly toast explains why it was rejected — and no modal pops."""
    frontend.coordinator._handle_online_error({"reason": "not_your_turn", "msg_type": msg_type})
    assert frontend.toast.is_visible()
    assert frontend.toast.message == NOT_YOUR_TURN_TOASTS[msg_type]
    assert not frontend.confirm_modal.is_visible()


def test_hard_failure_set_is_well_formed():
    assert "server_unreachable" in ONLINE_HARD_FAILURE_REASONS
    assert "reconnect_failed" in ONLINE_HARD_FAILURE_REASONS


def test_a_refused_request_tears_the_search_down_instead_of_spinning(frontend):
    """A body the server refuses used to reach the client as the generic
    `http_422`, which landed on this modal only by way of the `http_` prefix
    fallback -- so it read "Server unreachable" for a server that answered
    perfectly well. `invalid_field` is a first-class reason with its own label
    now, and being a hard failure it still takes the wait card down: a refused
    matchmake would otherwise leave the player watching a spinner for an opponent
    the server never queued."""
    frontend.coordinator.wait_modal.show("Rapid", "10 + 5", on_cancel=lambda: None)

    frontend.coordinator._handle_online_error({"reason": Reason.INVALID_FIELD})

    assert frontend.confirm_modal.is_visible()
    assert frontend.confirm_modal.title == "Request rejected by server"
    assert not frontend.coordinator.wait_modal.is_visible()
    assert frontend.toast.is_visible() is False


def test_menu_mode_skips_board_draw(frontend, monkeypatch):
    drew = []
    monkeypatch.setattr(frontend.game.board, "draw_board",
                        lambda: drew.append(True))
    frontend.switch_to("menu")
    frontend.draw_frame()
    assert drew == []
    frontend.switch_to("game", mode="single_screen")
    frontend.draw_frame()
    assert drew == [True]


def test_play_hero_lays_out_open_in_the_hero_column(frontend):
    """No card: the hero content lays out open across its column, left-aligned and
    clearing the left rail, with the CTA spanning the full column width."""
    frontend.draw_frame()
    layout = frontend.menu._menu_layout
    hero = frontend.menu.play_view
    assert layout.hero_rect.left >= layout.rail_rect.right
    assert hero._title_pos[0] == layout.hero_rect.x
    assert hero._cta_rect.x == layout.hero_rect.x
    assert hero._cta_rect.width == layout.hero_rect.width
    assert hero._cta_rect.bottom <= layout.hero_rect.bottom


def test_menu_mode_centers_flex_modals_on_window(frontend):
    assert frontend.screen is frontend.menu
    frontend._compute_layout()
    win_w, _ = frontend.window.get_size()
    assert abs(frontend.menu.fen_input_modal.rect.centerx - win_w / 2) <= 1
    assert abs(frontend.coordinator.wait_modal.rect.centerx - win_w / 2) <= 1
    assert abs(frontend.coordinator.match_found_modal.rect.centerx - win_w / 2) <= 1


def _board_centerx(board):
    return board.board_offset_x + board.cell_size * BOARD_SIZE / 2


def test_game_mode_centers_flex_modals_on_board(frontend):
    frontend.switch_to("game")
    frontend._compute_layout()
    board_cx = _board_centerx(frontend.game.board)
    assert abs(frontend.coordinator.wait_modal.rect.centerx - board_cx) <= 4
    assert abs(frontend.coordinator.match_found_modal.rect.centerx - board_cx) <= 4


def test_mode_change_relays_modal_rects_via_draw_frame(frontend):
    frontend.switch_to("menu")
    win_w, _ = frontend.window.get_size()
    assert abs(frontend.coordinator.wait_modal.rect.centerx - win_w / 2) <= 1
    frontend.switch_to("game")
    frontend.draw_frame()
    board_cx = _board_centerx(frontend.game.board)
    assert abs(frontend.coordinator.wait_modal.rect.centerx - board_cx) <= 4


def test_rematch_request_shows_banner_and_hides_initiate_button(frontend):
    """An incoming rematch request pushes a persistent Accept/Deny banner and
    flags the result modal to drop its initiate button (the banner is the affordance)."""
    sent = []
    frontend.coordinator.client = SimpleNamespace(
        send_rematch_response=lambda accept: sent.append(accept),
        send_rematch_request=lambda: sent.append("req"),
    )
    frontend.game._chosen_side = "white"
    frontend.game.white_name, frontend.game.black_name = "Me", "Them"
    frontend.coordinator._handle_rematch_request()
    assert frontend.coordinator._rematch_offered is True
    assert frontend.game.result_menu.rematch_offered is True
    assert not frontend.coordinator.offer_banners.is_empty()


def test_draw_offer_banner_pops(frontend):
    frontend.sound_manager = MagicMock()
    frontend.coordinator.client = SimpleNamespace(send_draw_response=lambda v: None)
    frontend.coordinator._push_offer_banner("draw_offered")
    frontend.sound_manager.play_toast.assert_called_once()


def test_takeback_offer_banner_pops(frontend):
    frontend.sound_manager = MagicMock()
    frontend.coordinator.client = SimpleNamespace(send_takeback_response=lambda v: None)
    frontend.coordinator._push_offer_banner("takeback_offered")
    frontend.sound_manager.play_toast.assert_called_once()


def test_offer_accept_and_decline_pop_and_send(frontend):
    frontend.sound_manager = MagicMock()
    sent = []
    frontend.coordinator.client = SimpleNamespace(send_draw_response=sent.append)
    captured = {}
    frontend.coordinator.offer_banners = MagicMock()
    frontend.coordinator.offer_banners.push = (
        lambda *a, on_yes, on_no, **k: captured.update(ok=on_yes, no=on_no))
    frontend.coordinator._push_offer_banner("draw_offered")
    frontend.sound_manager.reset_mock()
    captured["ok"]()
    frontend.sound_manager.play_toast.assert_called_once()
    assert sent == [True]
    frontend.sound_manager.reset_mock()
    captured["no"]()
    frontend.sound_manager.play_toast.assert_called_once()
    assert sent == [True, False]


def test_on_rematch_accepts_when_offered(frontend):
    sent = []
    frontend.coordinator.client = SimpleNamespace(
        send_rematch_response=lambda accept: sent.append(("resp", accept)),
        send_rematch_request=lambda: sent.append(("req",)))
    frontend.coordinator._rematch_offered = True
    frontend.game.result_menu.set_rematch_offered(True)
    frontend.coordinator._on_rematch()
    assert sent == [("resp", True)]
    assert frontend.coordinator._rematch_offered is False
    assert frontend.game.result_menu.rematch_offered is False


def test_on_rematch_requests_when_not_offered(frontend):
    sent = []
    frontend.coordinator.client = SimpleNamespace(
        send_rematch_response=lambda accept: sent.append(("resp", accept)),
        send_rematch_request=lambda: sent.append(("req",)))
    frontend.coordinator._rematch_offered = False
    frontend.coordinator._on_rematch()
    assert sent == [("req",)]


def _arm_post_game(frontend, **client):
    frontend.coordinator.client = SimpleNamespace(**client)
    frontend.game.variant = "online"
    frontend.game.manual_result = "draw_agreement"
    frontend.game._chosen_side = "white"
    frontend.game.white_name, frontend.game.black_name = "Me", "Them"
    frontend.menu.hide_play_view()


def test_decline_rematch_sends_false_and_clears(frontend):
    sent = []
    frontend.coordinator.client = SimpleNamespace(
        send_rematch_response=lambda accept: sent.append(accept))
    frontend.coordinator._rematch_offered = True
    frontend.game.result_menu.set_rematch_offered(True)
    frontend.coordinator._decline_rematch()
    assert sent == [False]
    assert frontend.coordinator._rematch_offered is False
    assert frontend.game.result_menu.rematch_offered is False


def test_rematch_update_reconnecting_keeps_client(frontend):
    _arm_post_game(frontend, disconnect=lambda: None)
    frontend.coordinator._handle_rematch_update({"event": "opponent_reconnecting"})
    assert frontend.toast.is_visible()
    assert frontend.coordinator.client is not None
    assert frontend.game.variant == "online"


def test_rematch_update_declined_bubbles_and_returns_to_menu(frontend):
    """The declined/opponent-left/window-expired paths must re-show the start-menu
    card; the old guard left the player on a blank, unresponsive backdrop."""
    _arm_post_game(frontend, disconnect=lambda: None)
    frontend.coordinator._rematch_offered = True
    frontend.coordinator._handle_rematch_update({"event": "declined"})
    assert frontend.toast.is_visible()
    assert frontend.coordinator.client is None
    assert frontend.screen is frontend.menu
    assert frontend.coordinator._rematch_offered is False
    assert frontend.menu.play_view_visible()


def test_rematch_update_window_expired_returns_to_menu(frontend):
    _arm_post_game(frontend, disconnect=lambda: None)
    frontend.coordinator._handle_rematch_update({"event": "window_expired"})
    assert frontend.coordinator.client is None
    assert frontend.screen is frontend.menu
    assert frontend.menu.play_view_visible()


def test_rematch_update_declined_logs_the_teardown_reason(frontend, caplog):
    _arm_post_game(frontend, disconnect=lambda: None)
    frontend.coordinator._rematch_offered = True
    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        frontend.coordinator._handle_rematch_update({"event": "declined"})
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("online session teardown")]
    assert lines == ["online session teardown reason=rematch_window_closed"]


def test_abandon_online_game_logs_the_teardown_reason(frontend, caplog):
    frontend.coordinator.client = SimpleNamespace(disconnect=lambda: None)
    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        frontend.coordinator._abandon_online_game()
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("online session teardown")]
    assert lines == ["online session teardown reason=reconnect_cancelled"]


def test_restart_online_search_logs_the_teardown_reason(frontend, caplog):
    frontend.coordinator.client = SimpleNamespace(disconnect=lambda: None)
    frontend._online_config = None
    with caplog.at_level(logging.INFO, logger="chess.frontend"):
        frontend.coordinator._restart_online_search()
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("online session teardown")]
    assert lines == ["online session teardown reason=restart_search"]


def test_online_result_redelivery_does_not_double_count(frontend, monkeypatch):
    """A reconnect re-delivers the result; the handler must be idempotent so the
    series score (and PGN auto-save) only fire once per game."""
    monkeypatch.setattr(frontend.game.result_flow, "auto_save_pgn", lambda: None)
    frontend.game.variant = "online"
    frontend.game._chosen_side = "white"
    frontend.game.white_name, frontend.game.black_name = "Me", "Them"
    frontend.game.result_flow.series_scores = {}
    frontend.game.manual_result = None
    payload = {"reason": "resignation", "winner_color": "white"}
    frontend.coordinator._handle_online_result(payload)
    frontend.coordinator._handle_online_result(payload)
    assert frontend.game.result_flow.series_scores["Me"] == 1.0
    assert frontend.game.manual_result == "white_wins_by_resignation"


def test_starting_fen_game_drops_lingering_online_client(frontend):
    """A kept-alive post-game online socket must be torn down when a local game
    starts, or its rematch banner / result leaks over the local game."""
    disc = []
    frontend.coordinator.client = SimpleNamespace(disconnect=lambda: disc.append(True))
    assert frontend.screen is frontend.menu
    ok = frontend._start_game_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert ok is True
    assert disc == [True]
    assert frontend.coordinator.client is None


def test_rematch_update_cancelled_clears_banner_and_keeps_client(frontend):
    _arm_post_game(frontend, disconnect=lambda: None)
    frontend.coordinator._push_rematch_banner()
    frontend.game.result_menu.set_rematch_offered(True)
    assert not frontend.coordinator.offer_banners.is_empty()
    frontend.coordinator._handle_rematch_update({"event": "cancelled"})
    assert frontend.coordinator.offer_banners.is_empty()
    assert frontend.coordinator._rematch_offered is False
    assert frontend.game.result_menu.rematch_offered is False
    assert frontend.toast.is_visible()
    assert frontend.coordinator.client is not None
    assert frontend.game.variant == "online"


def test_back_to_menu_keeps_client_and_reshows_banner_post_game(frontend):
    sent = []
    _arm_post_game(
        frontend,
        send_left_result=lambda: sent.append("left"),
        disconnect=lambda: sent.append("disc"),
    )
    frontend.coordinator._rematch_offered = True
    frontend._on_back_to_menu()
    assert sent == ["left"]
    assert frontend.coordinator.client is not None
    assert frontend.screen is frontend.menu
    assert not frontend.coordinator.offer_banners.is_empty()


def test_match_found_rematch_uses_rematch_eyebrow(frontend):
    frontend.coordinator.match_found_modal.show(
        "A", "B", "white", lambda: None, seconds=3, rematch=True)
    frontend.coordinator.match_found_modal.window.fill((0, 0, 0))
    frontend.coordinator.match_found_modal.draw()
    assert frontend.coordinator.match_found_modal.rematch is True
