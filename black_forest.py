class NvidiaImageGenerationProvider(ImageGenerationProvider):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)
        self.seed = 42
        self.cover_size = 512
        self.steps = 50
        self.timeout = 60
        self.transport = httpx.HTTPTransport(retries=3)

        self.api_key = settings.nvidia_api_key
        base = settings.nvidia_image_base_url.rstrip("/")
        self.model = settings.nvidia_image_model.strip()
        self.invoke_url = f"{base}/{self.model}"

    def generate_image(self, prompt: str, output_path: str) -> str:
        payload = {
            "prompt": prompt,
            "width": self.cover_size,
            "height": self.cover_size,
            # A fixed seed keeps a re-run of the same brief stable.
            "seed": self.seed,
            "steps": self.steps,
        }

        try:
            with httpx.Client(
                timeout=self.timeout, transport=self.transport  # type: ignore[arg-type]
            ) as client:
                response = client.post(
                    self.invoke_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ImageProviderError(
                f"NVIDIA image generation could not be reached: {exc}"
            ) from exc
        if response.status_code == 401:
            raise ImageProviderError("NVIDIA rejected the API key for image generation.")
        if response.status_code == 429:
            raise ImageProviderError("NVIDIA is rate limiting image generation. Try again shortly.")

        try:
            body = response.json()
        except ValueError as exc:
            raise ImageProviderError("NVIDIA returned a response that was not JSON.") from exc  

        # print(body)
        # Handle this error: {'detail': [{'type': 'string_too_long', 'loc': ['body', 'prompt'], 'msg': 'String should have at most 2000 characters'}]}

        base64_image = body.get("artifacts", [{}])[0].get("base64")

        try:
            data = base64.b64decode(base64_image, validate=True)
        except (ValueError, TypeError) as exc:
            raise ImageProviderError("NVIDIA returned an image that could not be decoded.") from exc
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path