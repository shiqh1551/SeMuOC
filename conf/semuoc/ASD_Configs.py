class Config(object):
    def __init__(self):
        
        self.dataset = 'ASD'
        self.input_channels = 19
        self.window_size = 32
        self.time_step = 16

        
        self.d_model = 96
        self.tcn_layers = 3
        self.attention_heads = 8
        self.dropout = 0.2
        self.score_alpha = 1.0
        self.score_beta = 0.5
        self.text_model = "intfloat/e5-base-v2"
        self.allow_hash_text_encoder = False
        self.text_embedding_cache = True
        self.text_cache_on_device = True
        self.domain_text = (
            "ASD is a multivariate application service monitoring dataset. "
            "Variables describe service-level runtime behavior, resource usage, "
            "and operational indicators. Normal behavior should preserve temporal "
            "regularity and cross-variable consistency."
        )
        self.variable_desc = {
            f"Channel-{idx}": f"ASD monitoring variable {idx}"
            for idx in range(self.input_channels)
        }
        self.anomaly_texts = [
            "Anomaly pattern: an application service variable suddenly spikes or drops within a short window.",
            "Anomaly pattern: a resource or workload indicator drifts away from its normal operating level.",
            "Anomaly pattern: normally correlated service variables become inconsistent with each other.",
            "Anomaly pattern: an expected periodic runtime pattern disappears or changes frequency.",
            "Anomaly pattern: a previously smooth service variable becomes highly volatile with local bursts.",
            "Anomaly pattern: multiple application indicators show conflicting runtime states.",
        ]

        
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

        
        self.num_epoch = 10
        self.early_stop = 20
        self.save_diagnostics = True
        self.keep_checkpoints = True

        
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-4

        
        self.batch_size = 128
        self.drop_last = False

        
        self.detect_nu = 0.0005
        self.threshold_determine = 'floating'
