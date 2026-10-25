import asyncio

import httpx

from revenue_analysis.ml import FEATURES, train
from revenue_analysis import service


def test_inference_validates_and_scores(fixture_frame, tmp_path, monkeypatch):
    train(fixture_frame, tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    service.startup()
    record = fixture_frame.loc[0, FEATURES].where(fixture_frame.loc[0, FEATURES].notna(), None).to_dict()

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test") as client:
            assert (await client.get("/health")).json()["status"] == "ok"
            response = await client.post("/predict", json={"records": [record]})
            assert response.status_code == 200, response.text
            assert 0 <= response.json()["p_retained"][0] <= 1
            assert "drift" in response.json()
            assert (await client.post("/predict", json={"records": [{**record, "churned": 1}]})).status_code == 422
            assert (await client.post("/predict", json={"records": [{**record, "price": -1}]})).status_code == 422

    asyncio.run(check())
