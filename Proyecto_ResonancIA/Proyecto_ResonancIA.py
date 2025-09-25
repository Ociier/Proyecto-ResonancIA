import reflex as rx
import asyncio
import os
import logging

from rxconfig import config

UPLOAD_ID = "image_upload"

class ImageUploadState(rx.State):
    """Manages the state for the image uploader application."""

    uploaded_images: list[str] = []
    is_uploading: bool = False
    upload_progress: int = 0

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """
        Handles the file upload process.

        Args:
            files: A list of files to be uploaded.
        """
        if not files:
            yield rx.toast.error("Please select at least one image to upload.")
            return
        self.is_uploading = True
        yield
        for i, file in enumerate(files):
            try:
                upload_data = await file.read()
                file_path = rx.get_upload_dir() / file.name
                with file_path.open("wb") as f:
                    f.write(upload_data)
                if file.name not in self.uploaded_images:
                    self.uploaded_images.append(file.name)
                self.upload_progress = int((i + 1) / len(files) * 100)
                await asyncio.sleep(0.1)
                yield
            except Exception as e:
                logging.exception(f"Error uploading {file.name}: {e}")
                self.is_uploading = False
                yield rx.toast.error(f"Error uploading {file.name}: {str(e)}")
                return
        self.is_uploading = False
        self.upload_progress = 0
        yield rx.toast.success(f"Successfully uploaded {len(files)} image(s)!")

    @rx.event
    def delete_image(self, filename: str):
        """
        Deletes a specific uploaded image.

        Args:
            filename: The name of the file to delete.
        """
        try:
            file_path = rx.get_upload_dir() / filename
            if file_path.exists():
                os.remove(file_path)
            self.uploaded_images.remove(filename)
            yield rx.toast.info(f"Image '{filename}' deleted.")
        except Exception as e:
            logging.exception(f"Failed to delete {filename}: {e}")
            yield rx.toast.error(f"Failed to delete {filename}: {str(e)}")

class State(rx.State):
    """The app state."""

def upload_component() -> rx.Component:
    """The component for uploading images."""
    return rx.el.div(
        rx.upload.root(
            rx.el.div(
                rx.icon("image-up", class_name="w-16 h-16 text-blue-500/50"),
                rx.el.h3(
                    "Drag and drop files here or click to select",
                    class_name="mt-4 text-base font-semibold text-gray-700",
                ),
                rx.el.p(
                    "Supports: .png, .jpg, .jpeg",
                    class_name="mt-1 text-sm text-gray-500",
                ),
                class_name="flex flex-col items-center justify-center p-8 sm:p-12 border-2 border-dashed border-gray-300 rounded-2xl bg-gray-50 hover:bg-blue-50 transition-colors duration-300 cursor-pointer",
            ),
            id=UPLOAD_ID,
            accept={"image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"]},
            multiple=True,
            max_files=10,
            class_name="w-full",
            on_drop=ImageUploadState.handle_upload(
                rx.upload_files(upload_id=UPLOAD_ID)
            ),
        ),
        rx.el.div(
            rx.cond(
                rx.selected_files(UPLOAD_ID).length() > 0,
                rx.el.div(
                    rx.el.div(
                        rx.foreach(
                            rx.selected_files(UPLOAD_ID),
                            lambda file: rx.el.div(
                                rx.icon(
                                    "file-image", class_name="w-5 h-5 text-blue-600"
                                ),
                                rx.el.span(file, class_name="truncate"),
                                class_name="flex items-center gap-2 p-2 bg-blue-100/50 border border-blue-200/50 rounded-lg text-sm text-gray-800 font-medium",
                            ),
                        ),
                        class_name="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3",
                    ),
                    rx.el.div(
                        rx.el.button(
                            "Clear Selection",
                            on_click=rx.clear_selected_files(UPLOAD_ID),
                            class_name="px-4 py-2 text-sm font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-all",
                        ),
                        rx.el.button(
                            rx.cond(
                                ImageUploadState.is_uploading,
                                rx.el.div(
                                    rx.icon("loader", class_name="animate-spin mr-2"),
                                    "Uploading...",
                                ),
                                "Upload Files",
                            ),
                            on_click=ImageUploadState.handle_upload(
                                rx.upload_files(upload_id=UPLOAD_ID)
                            ),
                            disabled=ImageUploadState.is_uploading,
                            class_name="px-5 py-2 text-sm font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-all flex items-center disabled:bg-blue-400 disabled:cursor-not-allowed",
                        ),
                        class_name="flex items-center justify-end gap-3 mt-4",
                    ),
                    class_name="w-full p-4 mt-4 bg-white border border-gray-200 rounded-xl shadow-sm",
                ),
            )
        ),
        rx.cond(
            ImageUploadState.is_uploading,
            rx.el.div(
                rx.el.progress(
                    value=ImageUploadState.upload_progress,
                    max=100,
                    class_name="w-full h-2 [&::-webkit-progress-bar]:rounded-full [&::-webkit-progress-value]:rounded-full [&::-webkit-progress-bar]:bg-slate-300 [&::-webkit-progress-value]:bg-blue-600 [&::-moz-progress-bar]:bg-blue-600",
                ),
                rx.el.p(
                    f"{ImageUploadState.upload_progress}%",
                    class_name="text-sm font-medium text-blue-600 mt-2 text-center",
                ),
                class_name="w-full mt-4",
            ),
        ),
        class_name="w-full max-w-2xl mx-auto flex flex-col items-center",
    )

def gallery_component() -> rx.Component:
    """Displays the gallery of uploaded images."""
    return rx.el.div(
        rx.cond(
            ImageUploadState.uploaded_images.length() > 0,
            rx.el.div(
                rx.el.h2(
                    "Your Uploaded Images",
                    class_name="text-2xl font-bold text-gray-800 mb-6",
                ),
                rx.el.div(
                    rx.foreach(ImageUploadState.uploaded_images, image_card),
                    class_name="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4",
                ),
                class_name="w-full",
            ),
            rx.el.div(
                rx.icon("image-off", class_name="w-20 h-20 text-gray-300"),
                rx.el.p(
                    "No images uploaded yet.",
                    class_name="mt-4 text-lg font-medium text-gray-500",
                ),
                class_name="w-full flex flex-col items-center justify-center p-16 bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200",
            ),
        ),
        class_name="w-full max-w-6xl mx-auto mt-12",
    )

def image_card(filename: str) -> rx.Component:
    """A card that displays an image and a delete button."""
    return rx.el.div(
        rx.el.image(
            src=rx.get_upload_url(filename),
            alt=f"Uploaded image: {filename}",
            class_name="aspect-square w-full object-cover rounded-lg transition-transform duration-300 group-hover:scale-105",
        ),
        rx.el.div(
            rx.el.p(filename, class_name="text-xs font-medium text-gray-700 truncate"),
            class_name="absolute bottom-0 left-0 right-0 p-2 bg-white/70 backdrop-blur-sm rounded-b-lg",
        ),
        rx.el.button(
            rx.icon("x", class_name="w-4 h-4"),
            on_click=lambda: ImageUploadState.delete_image(filename),
            class_name="absolute top-2 right-2 p-1.5 bg-black/40 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:bg-red-500 hover:scale-110",
            aria_label="Delete image",
        ),
        class_name="relative group overflow-hidden rounded-xl shadow-md border border-gray-200/50 hover:shadow-xl transition-shadow duration-300",
    )

def index() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("ResonancIA", size="9"),
            rx.text(
                "Para empezar, sube una resonancia válida.",
                size="5",
            ),
            rx.el.div(upload_component(), class_name="mt-10"),
            gallery_component(),
            rx.link(
                rx.button("Test"),
                href="https://reflex.dev/docs/getting-started/introduction/",
                is_external=True,
            ),
            spacing="5",
            justify="center",
            min_height="85vh",
        ),
    )


app = rx.App()
app.add_page(index)
