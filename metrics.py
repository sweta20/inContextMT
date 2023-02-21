import sacrebleu
import sys
from bert_score import BERTScorer
import numpy as np
from comet import download_model, load_from_checkpoint

class Metric():
  def __init__(self, lang="en"):
    self.name = ""
    self.lang = lang

  def _get_score(self, references, outputs):
    """Add scoring method here"""
    pass

  def get_score(self, references, outputs):
    """
    Args:
      reference: list of reference translations
      output: list of system outputs
    Returns:
      first value: corpus level score
      second value: list of sentence level scores
    """
    return self._get_score(references, outputs)

class BERTScoreMetric(Metric):
  def __init__(self, lang="en"):
    super().__init__(lang)
    self.name = "BERTScore"
    self.scorer = BERTScorer(lang=self.lang)

  def _get_score(self, references, outputs):
    P, R, F1 = self.scorer.score(outputs, references)
    return F1.numpy().mean(), F1.numpy()

class BleuMetric(Metric):
  def __init__(self, lang="en"):
    super().__init__(lang)
    self.name = "BLEU"
    self.lang = lang

  def _get_score(self, references, outputs):
    return sacrebleu.corpus_bleu(outputs, [references]), [sacrebleu.sentence_bleu(x,[y]).score for x, y in zip(outputs, references)]

class ChrfMetric(Metric):
  def __init__(self, lang="en"):
    super().__init__(lang)
    self.name = "CHRF"
    self.lang = lang

  def _get_score(self, references, outputs):
    return sacrebleu.corpus_chrf(outputs, [references]), [sacrebleu.sentence_chrf(x,[y]).score for x, y in zip(outputs, references)]

class COMETSrcMetric():
  def __init__(self, model_path="wmt21-comet-qe-mqm"):
    self.name = "COMETSrcScore"
    checkpoint_path = download_model(model_path)
    self.model = load_from_checkpoint(checkpoint_path)

  def get_score(self, references, outputs):
    data = [{"src": src, "mt": mt } for (src, mt) in zip(references, outputs)]
    seg_scores, _ = self.model.predict(data, batch_size=8, gpus=1)
    return np.mean(seg_scores), seg_scores


class COMETRefMetric():
  def __init__(self, model_path="eamt22-cometinho-da"):
    self.name = "COMETRefScore"
    checkpoint_path = download_model(model_path)
    self.model = load_from_checkpoint(checkpoint_path)

  def get_score(self, references, outputs, sources):
    data = [{"src": src, "mt": mt, "ref": ref} for (src, ref, mt) in zip(sources, references, outputs)]
    seg_scores, _ = self.model.predict(data, batch_size=8, gpus=1)
    return np.mean(seg_scores), seg_scores