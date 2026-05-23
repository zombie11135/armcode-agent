# core/context.py

class RuntimeContext:
    def __init__(
        self,
        camera=None,
        vlm=None,
        detector=None,
        segmenter=None,
        ocr=None,
        anygrasp=None,
        xarm=None,
        robot=None,
        event_callback=None,
    ):
        self.camera = camera
        self.vlm = vlm
        self.detector = detector
        self.segmenter = segmenter
        self.ocr = ocr
        self.anygrasp = anygrasp
        self.xarm = xarm
        self.robot = robot
        self.event_callback = event_callback

    def require(self, name: str):
        value = getattr(self, name, None)
        if value is None:
            raise RuntimeError(f"RuntimeContext missing required service: {name}")
        return value

    def emit(self, event: dict):
        if self.event_callback is not None:
            self.event_callback(event)

    def emit_text(self, content: str):
        self.emit({
            "type": "text",
            "content": content,
        })

    def emit_image(self, image_path: str, caption: str = ""):
        self.emit({
            "type": "image",
            "image_path": image_path,
            "content": caption,
        })

    def emit_error(self, content: str):
        self.emit({
            "type": "error",
            "content": content,
        })

    def emit_warning(self, content: str):
        self.emit({
            "type": "warning",
            "content": content,
        })