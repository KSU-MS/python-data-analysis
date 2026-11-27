"""Foxglove, live, for kmslap hil: a WebSocket server publishing the car (the same /sim
topics `kmslap mcap` writes) and the cell (/hil/...), with start / stop / reset / abort
as services, and the layout to watch it with (hil_layout).

Connect Foxglove to ws://localhost:8765 and import the layout (`kmslap hil` writes it).
"""
import json

import foxglove
import foxglove.layouts as fl
from foxglove.websocket import Capability, MessageSchema, Service, ServiceSchema
from mcap_protobuf.schema import build_file_descriptor_set

from kmslap.reporting.mcap_log import numeric_message

HIL_PACK_FIELDS = ["cell_voltage", "cell_current", "cell_temp", "soc", "cell_power_w", "pack_voltage", "pack_current",
                   "pack_power_kw", "cell_v_oc", "cell_r_mohm", "latency_ms", "lateness_ms", "rig_charge_ah",
                   "rig_energy_wh"]
HIL_RUN_FIELDS = ["state", "lap", "laps", "elapsed_s", "rig_state"]
RUN_STATES = {"idle": 0, "running": 1, "finished": 2, "fault": 3, "stopped": 4}
RIG_STATES = {"SIM": -1, "BOOT": 0, "IDLE": 1, "ARMED": 2, "RUNNING": 3, "FAULT": 4}


def hil_pack():
    return numeric_message("HilPack", HIL_PACK_FIELDS)


def hil_run():
    return numeric_message("HilRun", HIL_RUN_FIELDS)


_JSON_OBJECT = json.dumps({"type": "object"}).encode()


class Live:
    """The server. handlers: {service name: fn(request dict) -> response dict}, called
    from the server's threads."""

    def __init__(self, handlers: dict, *, host: str = "127.0.0.1", port: int = 8765):
        self._channels = {}
        services = []
        for name, fn in handlers.items():
            schema = ServiceSchema(
                f"kmslap.hil.{name}",
                request=MessageSchema(encoding="json", schema=foxglove.Schema(name=f"{name}_request", encoding="jsonschema",
                                                                               data=_JSON_OBJECT)),
                response=MessageSchema(encoding="json", schema=foxglove.Schema(name=f"{name}_response", encoding="jsonschema",
                                                                                data=_JSON_OBJECT)))
            services.append(Service(name, schema=schema, handler=self._wrap(fn)))
        self.server = foxglove.start_server(name="kmslap hil", host=host, port=port,
                                            capabilities=[Capability.Services], supported_encodings=["json"],
                                            services=services)
        self.port = port

    @staticmethod
    def _wrap(fn):
        def handler(request):
            try:
                body = json.loads(request.payload or b"{}") if request.payload else {}
                return json.dumps(fn(body) or {"ok": True}).encode()
            except Exception as e:          # the error goes back to the panel that called
                return json.dumps({"ok": False, "error": str(e)}).encode()
        return handler

    def declare(self, topic: str, cls):
        """Advertise a topic now (Foxglove shows it before anything is published)."""
        if topic not in self._channels:
            fds = build_file_descriptor_set(cls).SerializeToString()
            self._channels[topic] = foxglove.Channel(
                topic, schema=foxglove.Schema(name=cls.DESCRIPTOR.full_name, encoding="protobuf", data=fds),
                message_encoding="protobuf")

    def log(self, topic: str, pb, ns: int):
        self.declare(topic, type(pb))
        self._channels[topic].log(pb.SerializeToString(), log_time=ns)

    def close(self):
        self.server.stop()


# the SDK's panel types -> the ids the Foxglove app's layout files use
_CLASSIC_PANEL = {"ThreeDee": "3D", "Map": "map", "ServiceCall": "CallService"}


def classic_layout(sdk_layout: dict) -> dict:
    """A layout from foxglove.layouts (its `version: 1` form) in the form the Foxglove
    app imports from a file (configById + a binary split tree, as the team's own)."""
    config = {}
    count = iter(range(1, 10_000))

    def pid(kind):
        return f"{kind}!{next(count):x}"

    def node(n):
        if n["type"] == "panel":
            key = pid(_CLASSIC_PANEL.get(n["panelType"], n["panelType"]))
            config[key] = {**n.get("config", {}), **({"foxglovePanelTitle": n["title"]} if n.get("title") else {})}
            return key
        if n["type"] == "tabs":
            key = pid("Tab")
            config[key] = {"activeTabIdx": n.get("selectedTabIndex") or 0,
                           "tabs": [{"title": t["title"], "layout": node(t["content"])} for t in n["tabs"]]}
            return key
        items = n["items"]
        if len(items) == 1:
            return node(items[0]["content"])
        rest = {"type": "split", "direction": n["direction"], "items": items[1:]}
        total = sum(i.get("proportion", 1) for i in items)
        return {"first": node(items[0]["content"]), "second": node(rest), "direction": n["direction"],
                "splitPercentage": 100 * items[0].get("proportion", 1) / total}

    tree = node(sdk_layout["content"])
    return {"configById": config, "globalVariables": {}, "userNodes": {}, "playbackConfig": {"speed": 1},
            "layout": tree}


def hil_layout() -> str:
    """The layout to watch a run with: JSON to import in Foxglove (built with
    foxglove.layouts, written in the app's file format)."""
    def plot(title, *paths, y=None):
        return fl.PlotPanel(title=title, config=fl.PlotConfig(
            paths=[fl.PlotSeries(value=p, label=label) for p, label in paths], show_legend=True,
            legend_display="floating", time_window_mode="sliding", following_view_width=60,
            min_y_value=None if y is None else y[0], max_y_value=None if y is None else y[1]))

    def gauge(title, path, lo, hi):
        return fl.GaugePanel(title=title, config=fl.GaugeConfig(path=path, min_value=lo, max_value=hi,
                                                                 color_mode="colormap", color_map="turbo"))

    def button(service, text, color):
        return fl.ServiceCallPanel(title=text, config=fl.ServiceCallConfig(
            service_name=service, request_payload="{}", button_text=text, button_color=color, editing_mode=False))

    car3d = fl.ThreeDeePanel(title="car (isometric)", config=fl.ThreeDeeConfig(
        follow_tf="car", follow_mode="follow-position", fixed_frame="map",
        # the frames' axes and the map -> car line would cross the whole track
        scene=fl.BaseRendererSceneSettings(transforms=fl.BaseRendererTransforms(visible=False, show_label=False)),
        camera_state=fl.ThreeDeeCameraState(perspective=False, distance=14, phi=55, theta_offset=45,
                                            target_offset=(0, 0, 0)),
        topics={"/sim/track": fl.BaseRendererSceneUpdateTopicSettings(visible=True),
                "/sim/car": fl.BaseRendererSceneUpdateTopicSettings(visible=True)}))
    where = fl.MapPanel(title="where", config=fl.MapConfig(follow_topic="/sim/location", zoom_level=18))
    rig = fl.IndicatorPanel(title="rig", config=fl.IndicatorConfig(
        path="/hil/run.rig_state", style="background", fallback_label="no rig", rules=[
            fl.IndicatorRule(raw_value="-1", operator="=", color="#468ce6", label="SIM PACK"),
            fl.IndicatorRule(raw_value="1", operator="=", color="#888888", label="RIG IDLE"),
            fl.IndicatorRule(raw_value="2", operator="=", color="#e6aa32", label="RIG ARMED"),
            fl.IndicatorRule(raw_value="3", operator="=", color="#5abe6e", label="RIG RUNNING"),
            fl.IndicatorRule(raw_value="4", operator="=", color="#e6504a", label="RIG FAULT")]))
    run = fl.IndicatorPanel(title="run", config=fl.IndicatorConfig(
        path="/hil/run.state", style="background", fallback_label="no run", rules=[
            fl.IndicatorRule(raw_value="0", operator="=", color="#888888", label="IDLE"),
            fl.IndicatorRule(raw_value="1", operator="=", color="#5abe6e", label="RUNNING"),
            fl.IndicatorRule(raw_value="2", operator="=", color="#468ce6", label="FINISHED"),
            fl.IndicatorRule(raw_value="3", operator="=", color="#e6504a", label="FAULT"),
            fl.IndicatorRule(raw_value="4", operator="=", color="#e6aa32", label="STOPPED")]))
    controls = fl.SplitContainer(direction="row", items=[
        fl.SplitItem(proportion=1, content=button("start", "Start", "#5abe6e")),
        fl.SplitItem(proportion=1, content=button("stop", "Stop", "#e6aa32")),
        fl.SplitItem(proportion=1, content=button("reset", "Reset", "#468ce6")),
        fl.SplitItem(proportion=1, content=button("abort", "Abort", "#e6504a"))])
    gauges = fl.SplitContainer(direction="column", items=[
        fl.SplitItem(proportion=1, content=fl.SplitContainer(direction="row", items=[
            fl.SplitItem(proportion=1, content=gauge("cell current (A)", "/hil/pack.cell_current", -70, 160)),
            fl.SplitItem(proportion=1, content=gauge("cell voltage (V)", "/hil/pack.cell_voltage", 3.0, 4.3))])),
        fl.SplitItem(proportion=1, content=fl.SplitContainer(direction="row", items=[
            fl.SplitItem(proportion=1, content=gauge("cell temp (C)", "/hil/pack.cell_temp", 20, 60)),
            fl.SplitItem(proportion=1, content=gauge("SoC", "/hil/pack.soc", 0, 1))])),
        fl.SplitItem(proportion=1, content=fl.SplitContainer(direction="row", items=[
            fl.SplitItem(proportion=1, content=gauge("pack power (kW)", "/hil/pack.pack_power_kw", -40, 80)),
            fl.SplitItem(proportion=1, content=gauge("speed (m/s)", "/sim/state.speed", 0, 40))])),
        fl.SplitItem(proportion=0.5, content=fl.SplitContainer(direction="row", items=[
            fl.SplitItem(proportion=1, content=run), fl.SplitItem(proportion=1, content=rig)])),
        fl.SplitItem(proportion=0.5, content=controls)])
    tabs = fl.TabContainer(tabs=[
        fl.TabItem(title="cell", content=fl.SplitContainer(direction="column", items=[
            fl.SplitItem(proportion=1, content=plot("cell current (A)", ("/hil/pack.cell_current", "measured"))),
            fl.SplitItem(proportion=1, content=plot("cell voltage (V)", ("/hil/pack.cell_voltage", "measured"),
                                                    ("/hil/pack.cell_v_oc", "open circuit (fit)"))),
            fl.SplitItem(proportion=1, content=plot("cell temperature (C), SoC (%)", ("/hil/pack.cell_temp", "temp"))),
            fl.SplitItem(proportion=1, content=plot("resistance (mOhm)", ("/hil/pack.cell_r_mohm", "fit")))])),
        fl.TabItem(title="car", content=fl.SplitContainer(direction="column", items=[
            fl.SplitItem(proportion=1, content=plot("speed (m/s)", ("/sim/state.speed", "speed"))),
            fl.SplitItem(proportion=1, content=plot("pedals", ("/sim/state.pedal", "throttle"), ("/sim/state.brake", "brake"))),
            fl.SplitItem(proportion=1, content=plot("g", ("/sim/state.ax_g", "long"), ("/sim/state.ay_g", "lat"))),
            fl.SplitItem(proportion=1, content=plot("pack power (kW)", ("/hil/pack.pack_power_kw", "pack")))])),
        fl.TabItem(title="rig", content=fl.SplitContainer(direction="column", items=[
            fl.SplitItem(proportion=1, content=plot("loop latency (ms)", ("/hil/pack.latency_ms", "rig round trip"),
                                                    ("/hil/pack.lateness_ms", "sim behind real time"))),
            fl.SplitItem(proportion=1, content=plot("rig counters", ("/hil/pack.rig_charge_ah", "Ah"),
                                                    ("/hil/pack.rig_energy_wh", "Wh")))]))])
    layout = fl.Layout(content=fl.SplitContainer(direction="row", items=[
        fl.SplitItem(proportion=1.4, content=fl.SplitContainer(direction="column", items=[
            fl.SplitItem(proportion=2.2, content=car3d), fl.SplitItem(proportion=1, content=where)])),
        fl.SplitItem(proportion=1, content=gauges),
        fl.SplitItem(proportion=1.2, content=tabs)])).to_json()
    return json.dumps(classic_layout(json.loads(layout)), indent=1)
