import sys

if sys.version_info >= (3, 14):
    raise SystemExit(
        "This demo currently targets Python 3.12 and is not configured for Python 3.14+. "
        "Python 3.14 triggers a Pydantic v1 compatibility warning in the installed LangChain stack. "
        "Create and select a Python 3.12 virtual environment before starting the app."
    )

from travel_agent.interface.gradio_app import build_app


if __name__ == "__main__":
    build_app().queue().launch()
