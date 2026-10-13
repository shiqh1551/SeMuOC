class Config(object):
    def __init__(self):
        
        self.dataset = 'UCR'

        self.input_channels = 1
        self.window_size = 32
        self.time_step = 16

        
        self.d_model = 64
        self.tcn_layers = 3
        self.attention_heads = 4
        self.dropout = 0.1
        self.score_alpha = 1.0
        self.score_beta = 1.0
        self.text_model = "intfloat/e5-base-v2"
        self.allow_hash_text_encoder = False
        self.text_embedding_cache = True
        self.text_cache_on_device = True
        self.domain_text = (
            "UCR contains univariate time-series anomaly detection tasks from "
            "multiple domains. Normal behavior should preserve local smoothness, "
            "trend consistency, and expected periodic or shape patterns."
        )
        self.variable_desc = {
            f"Channel-{idx}": f"UCR time-series variable {idx}"
            for idx in range(self.input_channels)
        }
        self.anomaly_texts = [
            "Anomaly pattern: the univariate signal suddenly spikes or drops.",
            "Anomaly pattern: the signal drifts away from its normal trend.",
            "Anomaly pattern: the expected shape pattern changes unexpectedly.",
            "Anomaly pattern: periodic behavior disappears or shifts in frequency.",
            "Anomaly pattern: a smooth segment becomes volatile with a local burst.",
            "Anomaly pattern: a subsequence is inconsistent with the surrounding context.",
        ]

        # SemSphere one-class branch (AOC-style projection and fixed center)
        self.project_dim = 32
        self.oc_hidden_dim = None
        self.oc_pooling = "flatten"
        self.center_eps = 0.1
        self.center_normal_only = True

        
        self.use_infonce = True
        self.lambda_oc = 0.1
        self.lambda_align = 0.1
        self.lambda_anom = 0.1
        self.lambda_recon = 1.0
        self.anomaly_similarity_margin = 0.2

        
        self.num_epoch = 5
        self.early_stop = 20
        self.save_diagnostics = True
        self.keep_checkpoints = False

        
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-4

        
        self.batch_size = 512
        self.drop_last = False

        
        self.detect_nu = 0.0005
        self.threshold_determine = 'one-anomaly'
