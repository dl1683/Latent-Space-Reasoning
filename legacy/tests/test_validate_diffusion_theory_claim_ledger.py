from experiments.validate_diffusion_theory_claim_ledger import (
    parse_theory_claim_rows,
    validate_theory_claim_ledger,
)


def test_parse_theory_claim_rows_reads_claim_table(tmp_path):
    ledger = tmp_path / "ledger.md"
    ledger.write_text(_ledger(), encoding="utf-8")

    rows, issues = parse_theory_claim_rows(ledger)

    assert issues == []
    assert [row.claim_id for row in rows] == ["T1", "T2"]
    assert rows[0].status == "validated-local"
    assert "Run a check" in rows[0].next_proof_obligation


def test_validate_theory_claim_ledger_accepts_complete_rows_and_backlinks(tmp_path):
    ledger = tmp_path / "ledger.md"
    evidence = tmp_path / "evidence.md"
    backlink = tmp_path / "README.md"
    ledger.write_text(_ledger(evidence_ref="evidence.md"), encoding="utf-8")
    evidence.write_text("# Evidence\n", encoding="utf-8")
    backlink.write_text("See DIFFUSION_THEORY_CLAIM_LEDGER.md\n", encoding="utf-8")

    issues = validate_theory_claim_ledger(
        ledger_path=ledger,
        required_backlink_docs=(backlink,),
    )

    assert issues == []


def test_validate_theory_claim_ledger_resolves_archived_diffusion_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "docs" / "ledger.md"
    evidence = tmp_path / "docs" / "reports" / "diffusion" / "DIFFUSION_ARCHIVED_REPORT.md"
    backlink = tmp_path / "README.md"
    ledger.parent.mkdir(parents=True)
    evidence.parent.mkdir(parents=True)
    ledger.write_text(_ledger(evidence_ref="DIFFUSION_ARCHIVED_REPORT.md"), encoding="utf-8")
    evidence.write_text("# Evidence\n", encoding="utf-8")
    backlink.write_text("See DIFFUSION_THEORY_CLAIM_LEDGER.md\n", encoding="utf-8")

    issues = validate_theory_claim_ledger(
        ledger_path=ledger,
        required_backlink_docs=(backlink,),
    )

    assert issues == []


def test_validate_theory_claim_ledger_rejects_bad_status_missing_ref_and_weak_fields(tmp_path):
    ledger = tmp_path / "ledger.md"
    backlink = tmp_path / "README.md"
    ledger.write_text(
        "\n".join(
            [
                "# Ledger",
                "",
                "| ID | Status | Assertion | Current Evidence | Assumptions | Falsifier | Next Proof Obligation |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "| T1 | overclaimed | claim | `missing.md` | assumptions | never mind | maybe someday |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    backlink.write_text("No ledger mention\n", encoding="utf-8")

    issues = validate_theory_claim_ledger(
        ledger_path=ledger,
        required_backlink_docs=(backlink,),
    )

    reasons = {issue.reason for issue in issues}
    assert "unsupported status 'overclaimed'" in reasons
    assert "referenced evidence file does not exist: missing.md" in reasons
    assert "falsifier should name a disconfirming condition" in reasons
    assert "next proof obligation should name an action" in reasons
    assert any("required backlink doc does not mention" in reason for reason in reasons)


def test_validate_theory_claim_ledger_rejects_out_of_order_ids(tmp_path):
    ledger = tmp_path / "ledger.md"
    evidence = tmp_path / "evidence.md"
    backlink = tmp_path / "README.md"
    ledger.write_text(_ledger(first_id="T2", evidence_ref="evidence.md"), encoding="utf-8")
    evidence.write_text("# Evidence\n", encoding="utf-8")
    backlink.write_text("DIFFUSION_THEORY_CLAIM_LEDGER.md\n", encoding="utf-8")

    issues = validate_theory_claim_ledger(
        ledger_path=ledger,
        required_backlink_docs=(backlink,),
    )

    assert any(issue.reason == "expected ordered claim id T1" for issue in issues)


def _ledger(first_id="T1", evidence_ref="README.md"):
    return (
        "\n".join(
            [
                "# Ledger",
                "",
                "| ID | Status | Assertion | Current Evidence | Assumptions | Falsifier | Next Proof Obligation |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                f"| {first_id} | validated-local | First claim. | `{evidence_ref}` | Assumes evidence exists. | A fresh run fails to reproduce it. | Run a check on a held-out task. |",
                "| T2 | hypothesis | Second claim. | No file required. | Assumes transfer. | A held-out slice regresses. | Test the feature family. |",
            ]
        )
        + "\n"
    )
