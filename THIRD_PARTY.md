# Upstream provenance

This project builds on the following official repositories:

- https://github.com/explaining-markets/starter-modal — webhook lifecycle and deployment scaffold. The reviewed checkout was commit `12434cde288e24ec4dfd3e93a31be3b2edacd645`. `modal_app.py` is adapted from that scaffold; inference tests also originated from the earlier starter integration work. The `explaining_markets` SDK is installed separately rather than bundled here.
- https://github.com/explaining-markets/examples — archive loading, disclosure helpers and official scoring semantics. The reviewed scoring implementation was commit `501bd3182cd42cf61a727e05bd4b3089b8d1338d`; the experiments import the installed official package. The `examples` SDK is not bundled here.

Upstream authors retain their attribution and applicable license terms. No new blanket license is selected for this prepared repository. Review upstream notices and choose licensing for original contributions before presenting the project as generally reusable open-source software.

The archive and trained competition model are intentionally not distributed. Reproducing the reported archive experiment requires separately authorized data access.
