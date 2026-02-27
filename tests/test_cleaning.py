from cleaning.data_cleaning import clean_captions_file


def test_valid_entries_are_kept(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.jpg").write_bytes(b"fake")
    (imgs / "b.jpg").write_bytes(b"fake")

    captions = tmp_path / "captions.txt"
    captions.write_text("image,caption\na.jpg,a dog running\nb.jpg,a cat sleeping\n")

    out = tmp_path / "cleaned.txt"
    clean_captions_file(str(captions), str(out), str(imgs))

    text = out.read_text()
    assert "a.jpg" in text
    assert "b.jpg" in text


def test_missing_image_is_skipped(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "exists.jpg").write_bytes(b"fake")

    captions = tmp_path / "captions.txt"
    captions.write_text(
        "image,caption\nexists.jpg,present\nghost.jpg,missing image\n"
    )

    out = tmp_path / "cleaned.txt"
    clean_captions_file(str(captions), str(out), str(imgs))

    text = out.read_text()
    assert "exists.jpg" in text
    assert "ghost.jpg" not in text


def test_missing_captions_file_does_nothing(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    out = tmp_path / "cleaned.txt"

    clean_captions_file(str(tmp_path / "no_such_file.txt"), str(out), str(imgs))

    assert not out.exists()


def test_missing_images_folder_does_nothing(tmp_path):
    captions = tmp_path / "captions.txt"
    captions.write_text("image,caption\na.jpg,test\n")
    out = tmp_path / "cleaned.txt"

    clean_captions_file(str(captions), str(out), str(tmp_path / "no_folder"))

    assert not out.exists()


def test_existing_output_file_is_overwritten(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.jpg").write_bytes(b"fake")

    captions = tmp_path / "captions.txt"
    captions.write_text("header\na.jpg,first run\n")

    out = tmp_path / "cleaned.txt"
    out.write_text("old content from previous run")  # file already exists

    clean_captions_file(str(captions), str(out), str(imgs))

    text = out.read_text()
    assert "old content" not in text
    assert "a.jpg" in text


def test_all_images_missing_produces_no_output(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()  # empty — no images inside

    captions = tmp_path / "captions.txt"
    captions.write_text("header\nghost1.jpg,caption one\nghost2.jpg,caption two\n")

    out = tmp_path / "cleaned.txt"
    clean_captions_file(str(captions), str(out), str(imgs))

    assert not out.exists()


def test_invalid_format_lines_are_skipped(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (imgs / "a.jpg").write_bytes(b"fake")

    captions = tmp_path / "captions.txt"
    captions.write_text(
        "header\na.jpg,valid caption\nthis line has no comma\n\n"
    )

    out = tmp_path / "cleaned.txt"
    clean_captions_file(str(captions), str(out), str(imgs))

    text = out.read_text()
    assert "a.jpg" in text
    assert "this line has no comma" not in text
