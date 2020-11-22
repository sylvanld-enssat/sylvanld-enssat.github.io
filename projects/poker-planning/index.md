---
layout: article

title: Poker-Planning app
description: Help your team to put weight on agile project's tasks
sections: [3a, android, java]
---

> Simple poker planning app using Kotlin, databinding, and TCP/multicast sockets.

## Download release

### Multicast version
[multicast APK](../raw/master/releases/pokerplanning-multicast.apk)

[multicast AAB](../raw/master/releases/pokerplanning-multicast.aab)

### Broadcast version
[broadcast APK](../raw/master/releases/pokerplanning-broadcast.apk)

[broadcast AAB](../raw/master/releases/pokerplanning-broadcast.aab)

## Data class diagram
![data class diagram](docs/diagrams/class-diagram.png)

## Network communication

### Sessions discovery

![client list sessions](docs/diagrams/sessions-list.png)

![host start session](docs/diagrams/host-start-session.png)

### Join session

![client join session](docs/diagrams/join-session.png)

### Votes

![host request votes](docs/diagrams/vote-request.png)

![publish vote for feature](docs/diagrams/publish-vote-results.png)

### Summary

![session summary](docs/diagrams/publish-session-results.png)

