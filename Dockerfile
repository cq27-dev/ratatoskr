# The image a run's checks execute in.
#
# Not a build or release artifact — nothing ships from here. It exists so `[sandbox] backend =
# "container"` has somewhere to run that is not the host root, which is what stops a run reading
# `~/.ssh` and what lets `ratatoskr prepare` put dependencies where an offline check can find them.
#
# Built locally; there is no registry in the loop:
#
#     docker build -t ratatoskr-checks .
#
# Rebuild it when `rust-toolchain.toml` moves or the dashboard gains a toolchain need. Nothing
# rebuilds it for you, and a stale image runs the checks under a toolchain the repository no longer
# pins — which shows up as a lint that fires here and not in CI.

# bun rather than npm: the dashboard commits `bun.lock` and no `package-lock.json`, so bun is the
# only package manager here that can install from a lockfile rather than re-resolving ranges.
FROM docker.io/oven/bun:1 AS bun

# Pinned to `rust-toolchain.toml`. The image carries the toolchain so a run does not depend on
# whatever the operator's rustup happens to have selected.
FROM docker.io/library/rust:1.97.0-bookworm

# `libcap-ng-dev` is what CI installs before its checks: the optional `microsandbox` feature links
# the system libcap-ng through the `capng` crate. The default build does not need it, but the
# acceptance is derived from CI, so a run may well propose the feature-gated step.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcap-ng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun

# rustfmt and clippy are two of the three gates CI enforces, so an image without them can only ever
# run a third of the acceptance.
RUN rustup component add rustfmt clippy

# Where `cargo fetch` puts what it fetched, and therefore where `[[sandbox.cache]]` mounts it back.
# Stated here so the config's `at` and the image agree in one place rather than two.
ENV CARGO_HOME=/usr/local/cargo
