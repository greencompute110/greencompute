from greencompute_protocol.enums import DeploymentState
from greencompute_protocol.models import DeploymentStatusUpdate
from greencompute_protocol.ssh_host_keys import (
    MAX_KEYS,
    clean_ssh_host_keys,
    known_hosts_line,
    ssh_key_fingerprint,
)

# Made with ssh-keygen; the fingerprints are what `ssh-keygen -lf -E sha256` printed.
ED = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFDpfaWJh7dYs3/G3VaELhN36Tj0pY7oUj+Afdy7rk93"
ED_FP = "SHA256:A2j3vFHUT8XfIsjGuvojYy+ktwi5ULcpzurhUOD3jJg"
EC = ("ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEbBtMZtLYqFI5v+Dgiy8Hpi5QQd"
      "8pxTsAiqLATbh2bTDlNRKAcOaPO6FkAzthlAm4jQrj+KCUW2Ou65ZDcBAs8=")
EC_FP = "SHA256:6V4ELkVY8lIBJM0darySIIM0cKDEWZwJt8LW1sG65vw"


def test_reads_pub_files_and_drops_comments():
    text = f"{EC} root@abc123\n\n{ED} root@abc123\n"
    assert clean_ssh_host_keys(text) == [EC, ED]


def test_fingerprints_match_openssh():
    assert ssh_key_fingerprint(ED) == ED_FP
    assert ssh_key_fingerprint(EC) == EC_FP


def test_rejects_malformed_and_mislabelled_keys():
    ed_blob = ED.split()[1]
    assert clean_ssh_host_keys([
        "ssh-ed25519",                         # no key
        "ssh-dss " + ed_blob,                  # type we don't accept
        "ssh-ed25519 not*base64",              # not base64
        "ssh-rsa " + ed_blob,                  # blob says ed25519, label says rsa
        "ssh-ed25519 " + "A" * 3000,           # far too long
        "<script>alert(1)</script> " + ed_blob,
        42, None,
    ]) == []


def test_repeats_removed_and_capped():
    assert clean_ssh_host_keys([ED, ED + " other-comment", EC]) == [ED, EC]
    assert len(clean_ssh_host_keys([ED] * 50)) == 1
    many = [ED, EC] * 10
    assert len(clean_ssh_host_keys(many)) <= MAX_KEYS


def test_not_a_list_is_empty_not_an_error():
    assert clean_ssh_host_keys(None) == []
    assert clean_ssh_host_keys(12) == []


def test_known_hosts_line_matches_ssh_format():
    assert known_hosts_line("1.2.3.4", 2201, ED) == f"[1.2.3.4]:2201 {ED}"
    assert known_hosts_line("1.2.3.4", 22, ED) == f"1.2.3.4 {ED}"


def test_status_update_cleans_keys_and_keeps_none():
    update = DeploymentStatusUpdate(
        deployment_id="d", state=DeploymentState.READY, ssh_host_keys=[ED + " root@x", "junk"],
    )
    assert update.ssh_host_keys == [ED]
    assert DeploymentStatusUpdate(deployment_id="d", state=DeploymentState.READY).ssh_host_keys is None
    # A report from a machine that sends something odd still parses.
    assert DeploymentStatusUpdate.model_validate(
        {"deployment_id": "d", "state": "ready", "ssh_host_keys": "nonsense"}).ssh_host_keys == []
