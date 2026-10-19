# Publishing clips to YouTube Shorts, TikTok and Instagram Reels

clipper posts finished clips through each platform's **official API**. You create a free developer
app on each platform once, paste its keys into clipper, connect your account, and then every clip
gets a **Publish** button with Claude's post text filled in. You check and edit it, then post. No
clip is posted without you clicking Publish.

Your logins and keys stay on your Mac, in `~/.config/clipper/` (`apps.json`, `accounts.json`),
readable only by your user. Each post is recorded in `<work>/<video>/output/published.json`, and
clipper refuses to post the same clip to the same platform twice unless you ask it to.

> Only publish clips of videos you own or have permission to reuse. Re-uploading other creators'
> content can get your accounts struck or banned, and it breaks each platform's terms.

## What each platform allows for a personal developer app

| | Visibility of your posts | Extra requirements |
|---|---|---|
| **YouTube Shorts** | Public, unlisted or private, your choice | A YouTube channel. While the Google app is in "Testing", you reconnect about once a week. |
| **TikTok** | **Only you can see them** until TikTok audits your app; your account must be set to private while posting | TikTok's posting rules are built into the form (see below). Max 5 users a day before the audit. |
| **Instagram Reels** | Public | A Business or Creator Instagram account linked to a Facebook Page. |

Sign-in happens in your browser and returns to clipper at
`http://localhost:<port>/oauth/<platform>/callback`. The port is the one `clipper ui` prints
(8765 by default). Sign in from the computer running clipper, not from your phone.

## YouTube Shorts

1. Go to [console.cloud.google.com](https://console.cloud.google.com), create a project, and enable
   the **YouTube Data API v3** (APIs & Services → Library).
2. **OAuth consent screen**: choose *External*, fill in the app name and your email, add the scopes
   `.../auth/youtube.upload` and `.../auth/youtube.readonly`, and add your own Google account under
   **Test users**.
3. **Credentials → Create credentials → OAuth client ID → Desktop app**. Desktop clients accept any
   `http://localhost` port, so there's no redirect address to register.
4. In clipper: **Accounts → YouTube Shorts**, paste the **Client ID** and **Client secret**, then
   **Connect** and sign in with the Google account that owns your channel.

Vertical clips under 3 minutes show up as Shorts automatically. Each upload uses YouTube API quota,
which is generous for personal use.

## TikTok

1. Go to [developers.tiktok.com](https://developers.tiktok.com/apps), create an app, and choose the
   **Desktop** platform.
2. Add the products **Login Kit** and **Content Posting API**. In Content Posting API, turn on
   **Direct Post**.
3. Under Login Kit, add the redirect URI `http://localhost:8765/oauth/tiktok/callback`. If clipper
   ends up on another port, add that one too, or register `http://localhost:*/oauth/tiktok/callback`.
4. Request the scopes `user.info.basic` and `video.publish`.
5. In clipper: **Accounts → TikTok**, paste the **Client key** and **Client secret**, then **Connect**.
6. Before you post, set your TikTok account to **private** (required for unaudited apps).

To make posts public, submit the app for TikTok's audit in the developer portal.

The TikTok form follows TikTok's content-sharing rules, so it differs from the other two:

- it shows the TikTok account the clip goes to;
- you pick who can see the post: nothing is preselected, and only the options TikTok offers your
  account are listed;
- comments, Duet and Stitch are off until you turn them on, and an option is greyed out when your
  TikTok settings disable it;
- a **branded content** switch, where you say whether the clip promotes you, a third party, or both
  (branded content can't be private);
- you must agree to TikTok's Music Usage Confirmation before anything is uploaded.

## Instagram Reels

1. Make your Instagram account a **Business** or **Creator** account (Instagram → Settings →
   Account type) and **link it to a Facebook Page** you manage.
2. Go to [developers.facebook.com/apps](https://developers.facebook.com/apps) and create an app of
   type **Business**. Add **Facebook Login for Business** and the **Instagram Graph API**.
3. In Facebook Login settings, `http://localhost` redirect addresses are accepted while the app is in
   development mode. If Meta asks for one, add
   `http://localhost:8765/oauth/instagram/callback`.
4. Your own account has a role on the app, so you can use it in development mode without App Review.
5. In clipper: **Accounts → Instagram Reels**, paste the **App ID** and **App secret** (Settings →
   Basic). Add a **Configuration ID** only if you created a Facebook Login for Business
   configuration. Then **Connect** and allow access to your Page and Instagram account.

The login lasts about 60 days; clipper tells you when to reconnect. Instagram allows 100 posts per
account in 24 hours through the API.

## If something goes wrong

- **"isn't set up yet"**: the app keys for that platform are missing. Add them under Accounts.
- **"login has expired"**: click Connect again.
- **TikTok "only you can see it"**: expected until TikTok audits your app.
- **Instagram "No Instagram professional account is linked"**: check step 1 above.
- The Graph API version defaults to `v24.0`. Set `CLIPPER_META_GRAPH_VERSION` if Meta retires it.
