from bioteq.processing.base import Processor

class LocationProcessor(Processor):
    def process(self, readings):
        points = {}
        for r in readings:
            if r.value is not None:
                points.setdefault(r.timestamp_ms, {})[r.axis] = r.value
        complete = [(t, p) for t, p in points.items() if "latitude" in p and "longitude" in p]
        if not complete:
            return None
        t, point = max(complete)
        return {"timestamp_ms": t, **point}
