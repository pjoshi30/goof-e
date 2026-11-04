# Music

Goof-E plays music while the arm moves, to soften the sound of the motors
(inspired by Catie Cuan's "music mode" for robots).

Put one audio file here (MP3, M4A, WAV or OGG). The first file alphabetically is used.
Audio files in this folder are ignored by git, so the repo ships without one. With no
file, the web app plays a soft generated ambient pad instead.

Goof-E uses Erik Satie's *Gymnopédie No. 1*, which is public domain
([Wikimedia Commons recording](https://commons.wikimedia.org/wiki/File:Erik_Satie_-_gymnopedies_-_la_1_ere._lent_et_douloureux.ogg),
released by its performer Robin Alciatore via Musopen). To use it, download the file and
convert it to M4A so every browser can play it:

```bash
curl -L -o /tmp/gymnopedie.ogg "https://commons.wikimedia.org/wiki/Special:FilePath/Erik_Satie_-_gymnopedies_-_la_1_ere._lent_et_douloureux.ogg"
ffmpeg -i /tmp/gymnopedie.ogg -c:a aac -b:a 160k music/gymnopedie-no1-satie.m4a
```

Only add music you have the rights to use.
