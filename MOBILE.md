# Shipping VocalIQ to the App Store and Google Play

The apps are thin native shells (Capacitor) around the same web app in `web/`. Everything
the store apps need is already in place: a mobile-first UI, bearer-token login for apps,
CORS for app origins, microphone capture via WebView, local-notification reminders,
account deletion, a privacy-policy template, icons and splash artwork.

What's left needs your accounts and tools, so you do it once, in this order.

## 0 · Accounts and tools

| Need | For | Cost |
|---|---|---|
| [Apple Developer Program](https://developer.apple.com/programs/) | App Store | $99 / year |
| [Google Play Console](https://play.google.com/console) | Google Play | $25 once |
| A **Mac** with Xcode 16+ | building the iOS app (required by Apple) | |
| [Node.js 20+](https://nodejs.org) | Capacitor tooling | free |
| [Android Studio](https://developer.android.com/studio) | building the Android app (Windows is fine) | free |

## 1 · Host the server

Follow [DEPLOYMENT.md](DEPLOYMENT.md) until `https://YOUR-SERVER/api/health` responds.
The apps can't work without it.

## 2 · Choose your app ID

Edit `mobile/capacitor.config.json` → `appId`. It must be unique and can never change after
publishing. Reverse-domain style, e.g. `com.yourname.vocaliq`.

## 3 · Create the native projects (once)

```bash
cd mobile
npm install
VOCALIQ_API_BASE=https://YOUR-SERVER npm run sync      # copies ../web into www/, points it at your server
npx cap add android
npx cap add ios                                          # on the Mac
npm run assets                                           # icons + splash from mobile/resources/
```

(Windows PowerShell: `$env:VOCALIQ_API_BASE="https://YOUR-SERVER"; npm run sync`)

Commit the generated `mobile/android` and `mobile/ios` folders.

## 4 · Permissions (required, or the microphone won't work)

**iOS:** in `ios/App/App/Info.plist` add:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>VocalIQ listens while you sing so it can analyse your pitch, breath and tone.</string>
```

**Android:** in `android/app/src/main/AndroidManifest.xml`, inside `<manifest>`:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
```

Capacitor's web view asks for the microphone at runtime the first time a tool starts listening.

## 5 · Build, test, and upload

Every time the web app changes: `VOCALIQ_API_BASE=https://YOUR-SERVER npm run sync`.

- **Android:** `npm run android` opens Android Studio → run on a phone → *Build → Generate
  Signed App Bundle* → upload the `.aab` in Play Console (start with *Internal testing*).
- **iOS:** `npm run ios` opens Xcode → set your Team under *Signing & Capabilities* → run on an
  iPhone → *Product → Archive* → *Distribute App* → TestFlight → submit for review.

Test on real phones: recording, the tuner and warm-ups (with and without headphones),
reminders, share links, guest → account upgrade, and account deletion.

## 6 · Store listing checklist

- [ ] Privacy policy: fill in `web/privacy.html` (every `[placeholder]`) and host it; link it in both stores
- [ ] **Apple App Privacy** / **Google Data safety** forms: you collect *audio recordings*,
      *email address*, and *app activity*, used for app functionality, not tracking, not shared
- [ ] Account deletion: in-app at *Profile → Delete my account* (both stores require this)
- [ ] Screenshots: iPhone 6.9" and 6.5", iPad if supported, Android phone (the app's Today,
      report, tuner and progress screens make good ones)
- [ ] App icon is at `mobile/resources/icon.png` (1024×1024, no transparency)
- [ ] Age rating questionnaire; content is suitable for everyone
- [ ] Review notes for Apple: explain that "Try as guest" needs no sign-up so reviewers can test instantly
- [ ] Make sure reference-song uploads comply with copyright: users upload their own files,
      and the privacy policy already says the audio is deleted after comparison

## Known platform notes

- **iOS audio routing:** while the microphone is open, iOS can route sound to the earpiece.
  Warm-ups and pitch drills avoid this by closing the mic while the piano plays, but
  recommend headphones in the app listing.
- **Reminders** use `@capacitor/local-notifications` and only fire in the installed apps.
  On the website the time is saved and takes effect once the singer installs the app.
- **Offline:** the practice tools work offline in the apps; recording analysis and history need the server.
