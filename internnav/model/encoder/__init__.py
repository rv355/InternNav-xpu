from importlib import import_module


_ENCODERS = {
	'PositionalEncoding': '.bert_backbone',
	'DistanceNetwork': '.distance_encoder',
	'ImageEncoder': '.image_clip_encoder',
	'InstructionEncoder': '.instruction_encoder',
	'InstructionLongCLIPEncoder': '.instruction_longCLIP_encoder',
	'LanguageEncoder': '.instruction_roberta_encoder',
	'VisionLanguageEncoder': '.vision_language_encoder',
}
__all__ = list(_ENCODERS)


def __getattr__(name):
	if name not in _ENCODERS:
		raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
	encoder = getattr(import_module(_ENCODERS[name], __name__), name)
	globals()[name] = encoder
	return encoder
