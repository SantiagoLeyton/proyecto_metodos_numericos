from interfaz.gui_principal import create_app


def main() -> None:
    app = create_app()
    app.mainloop()


if __name__ == "__main__":
    main()