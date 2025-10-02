import reflex as rx

# Configura Reflex para tu proyecto
config = rx.Config(
    app_name="Proyecto_ResonancIA",
    plugins=[
        rx.Sitemap(),  # Usando el plugin de Sitemap directamente
        rx.TailwindV4(),  # Usando el plugin de Tailwind directamente
    ]
)
