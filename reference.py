from pathlib import Path
from io import BytesIO
import pandas as pd
import re
import boto3
from botocore.client import Config

S3_ENDPOINT = ""
S3_ACCESS_KEY = ""
S3_SECRET_KEY = ""
LAKE_BUCKET = ""

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)

# =========================================================
# Logical MinIO paths
# These are object keys / prefixes, not local filesystem paths
# =========================================================

# input area is at bucket root
INPUT_PREFIX = Path("input_data")

# governance area is under the project prefix
BASE_DIR = Path("well-being-and-age-friendliness")
GOV_DIR = BASE_DIR / "governance"
REF_DIR = GOV_DIR / "ref"

DATA_DIR = INPUT_PREFIX
METADATA_DIR = GOV_DIR / "metadata" / "raw"
CATALOG_PATH = GOV_DIR / "catalog.json"
PERF_DIR = GOV_DIR / "perf"
CACHE_DIR = GOV_DIR / "ref" / "_df_cache"

REF_SEMANTIC = REF_DIR / "ref_semantic"
REF_SPATIAL_DIR = REF_DIR / "ref_spatial"
DATASET_SOURCE_REGISTRY = REF_DIR / "dataset&source.xlsx"


def minio_key(path_obj) -> str:
    """Convert Path-like object to MinIO key string."""
    return str(path_obj).replace("\\", "/")

def read_csv_from_minio(path_obj, **kwargs) -> pd.DataFrame:
    """Read one CSV object directly from MinIO."""
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
    return pd.read_csv(BytesIO(obj["Body"].read()), **kwargs)

def read_excel_from_minio(path_obj, **kwargs) -> pd.DataFrame:
    """Read one Excel object directly from MinIO."""
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
    return pd.read_excel(BytesIO(obj["Body"].read()), **kwargs)

def read_json_from_minio(path_obj):
    """Read one JSON object directly from MinIO."""
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
    return obj["Body"].read()

def object_exists(path_obj) -> bool:
    """Check whether an object exists in MinIO."""
    try:
        s3.head_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
        return True
    except Exception:
        return False

DEFAULT_EXTS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".json",
    ".geojson",
    ".zip",
    ".shp",
}

pairs = ["reg", "academie", "dep", "arr_dep", "epci", "canton", "com", "com_arr", "iris"]

ref_dict = {
    p: read_csv_from_minio(REF_SPATIAL_DIR / f"{p}.csv", dtype=str) for p in pairs
}


HIER = {
    "spatial": [
        [
            "country",
            "region",
            "academie",
            "departement",
            "arrondissement_departemental",
            "canton",
            "commune",
            "arrondissement_communal",
            "iris",
            ("geometry", "wkt_geojson"),
            "latlon_pair",
            "address",
        ],
        [
            "country",
            "epci",
            "commune",
            "arrondissement_communal",
            "iris",
            ("geometry", "wkt_geojson"),
            "latlon_pair",
            "address",
        ],
        [
            "country",
            "insee_geo",
        ],
    ],
    "temporal": [["year", "quarter", "month", "date"],["year", "week", "date"]]
}

SPATIAL_NAME_MAP = {
    "reg": "region",
    "academie": "academie",
    "dep": "departement",
    "arr_dep": "arrondissement_departemental",
    "epci": "epci",
    "canton": "canton",
    "com": "commune",
    "com_arr": "arrondissement_communal",
    "iris": "iris",
    "geometry": "geometry",
    "wkt_geojson": "wkt_geojson",
    "latlon_pair": "latlon_pair",
    "address": "address",
}

THEME_FOLDER_STRUCTURE = {
        "Well-being": {
            "Current Well-being": {
                "Health": {
                    "Physical health": {
                        "Pain and discomfort": {},
                        "Energy and fatigue": {},
                        "Sleep and rest": {},
                        "Longevity & survival": {
                            "Life expectancy": {},
                            "Mortality": {}
                        },
                        "Disease burden": {
                            "Chronic conditions": {},
                            "Overweight & obesity": {},
                            "Risk behaviours": {
                                "Smoking & tobacco": {},
                                "Harmful alcohol use": {},
                                "Physical inactivity": {},
                                "Diet & nutrition": {}
                            }
                        }
                    },
                    "Mental health": {
                        "Positive feelings": {},
                        "Thinking, learning, memory and concentration": {
                            "Speed": {},
                            "Clarity": {}
                        },
                        "Self-esteem": {},
                        "Body image and appearance": {},
                        "Negative feelings": {},
                        "Mental state": {
                            "Emotional well-being": {},
                            "Cognitive function": {}
                        },
                        "Suicide & self-harm": {}
                    },
                    "Access to care": {
                        "Financial accessibility": {},
                        "Service availability": {
                            "Geographic accessibility": {},
                            "Coverage": {}
                        }
                    },
                    "Health systems & services": {
                        "Expenditure & financing": {},
                        "Workforce & resources": {},
                        "Utilisation & access": {},
                        "Quality & outcomes": {},
                        "Pharmaceuticals & medicines": {},
                        "Preventive services & screening": {},
                        "Health inequalities": {}
                    },
                    "Level of independence": {
                        "Mobility": {},
                        "Activities of daily living": {
                            "Taking care of oneself": {},
                            "Managing belongings appropriately": {}
                        },
                        "Dependence on medication and medical aids": {}
                    }
                },
                "Education & Skills": {
                    "Educational outcomes": {
                        "Attainment": {
                            "Years of schooling": {},
                            "Upper secondary attainment": {},
                            "Completion": {}
                        },
                        "Performance": {
                            "Literacy": {},
                            "Numeracy & science": {}
                        }
                    },
                    "Skills & learning": {
                        "Lifelong learning": {
                            "Adult education": {},
                            "Training opportunities": {}
                        },
                        "Skills level": {
                            "Digital skills": {},
                            "Employability skills": {}
                        }
                    }
                },
                "Income & Wealth": {
                    "Income": {
                        "Household income": {},
                        "Distribution": {
                            "Inequality": {},
                            "Poverty": {},
                            "Income inequality (Gini)": {},
                            "Relative poverty rate": {}
                        }
                    },
                    "Wealth": {
                        "Net wealth": {},
                        "Economic security": {
                            "Financial resilience": {},
                            "Perceived security": {},
                            "Feeling of having enough": {}
                        }
                    }
                },
                "Jobs & Earnings": {
                    "Employment quantity": {
                        "Participation": {
                            "Employment": {},
                            "Labour force participation": {}
                        },
                        "Unemployment": {
                            "General unemployment": {},
                            "Long-term unemployment": {}
                        }
                    },
                    "Job quality": {
                        "Wage level": {},
                        "Stability": {
                            "Contract type": {},
                            "Job security": {}
                        },
                        "Working conditions": {
                            "Occupational safety": {},
                            "Job satisfaction": {}
                        }
                    },
                    "Additional aspects": {
                        "Youth NEET rate": {},
                        "Informal employment rate": {},
                        "Work capacity": {}
                    }
                },
                "Housing": {
                    "Housing conditions": {
                        "Overcrowding": {},
                        "Facilities": {}
                    },
                    "Housing affordability": {
                        "Cost burden": {},
                        "Homelessness": {}
                    }
                },
                "Environment Quality": {
                    "Environmental exposure": {
                        "Air quality": {},
                        "Noise exposure": {}
                    },
                    "Perceptions & access": {
                        "Environmental satisfaction": {},
                        "Green space accessibility": {}
                    },
                    "Domestic environment": {
                        "Crowding": {},
                        "Available space": {},
                        "Cleanliness": {},
                        "Opportunities for privacy": {},
                        "Available equipment": {},
                        "Building construction quality": {}
                    },
                    "Basic services & utilities": {
                        "Transport": {},
                        "Drinking water": {},
                        "Gas": {},
                        "Electricity": {},
                        "Sewage networks": {}
                    },
                    "Urbanisation level": {},
                    "Comfort and security": {}
                },
                "Safety": {
                    "Personal safety": {
                        "Homicide and assault": {},
                        "Crime incidence": {},
                        "Perceived safety": {}
                    },
                    "Road safety": {
                        "Traffic injuries": {},
                        "Transport infrastructure safety": {}
                    }
                },
                "Civic Engagement & Governance": {
                    "Participation": {
                        "Electoral participation": {},
                        "Voter turnout": {},
                        "Civic participation (consultation, petitions)": {}
                    },
                    "Trust & satisfaction": {
                        "Institutional trust": {},
                        "Public service satisfaction": {},
                        "Access to justice": {},
                        "Perceived corruption": {}
                    }
                },
                "Social Connections": {
                    "Social support": {
                        "Reliance network": {},
                        "Help in times of need": {},
                        "Loneliness": {}
                    },
                    "Social participation": {
                        "Community participation": {},
                        "Informal care": {}
                    },
                    "Personal relations": {},
                    "Sexual activity": {}
                },
                "Subjective Well-being": {
                    "Life satisfaction": {},
                    "Affective balance": {
                        "Positive vs negative emotions": {},
                        "Positive affect": {},
                        "Negative affect": {}
                    }
                },
                "Work-life Balance": {
                    "Long working hours": {},
                    "Commuting time": {},
                    "Unpaid work": {},
                    "Leisure time": {},
                    "Childcare availability": {},
                    "Time use balance": {}
                },
                "Spirituality / Religion / Personal Beliefs": {}
            },
            "Resources for Future Well-being": {
                "Natural Capital": {
                    "Ecosystems & biodiversity": {
                        "Protected areas": {},
                        "Forest cover": {},
                        "Species conservation": {}
                    },
                    "Climate & sustainability": {
                        "Emissions": {},
                        "Renewable energy": {},
                        "Freshwater resources": {},
                        "Green & blue infrastructure": {}
                    }
                },
                "Human Capital": {
                    "Health stock": {
                        "Longevity": {},
                        "Child development": {}
                    },
                    "Education & skills stock": {
                        "Higher education": {},
                        "Foundational skills": {},
                        "Adult skills": {}
                    }
                },
                "Social Capital": {
                    "Trust & norms": {
                        "Interpersonal trust": {},
                        "Institutional trust": {}
                    },
                    "Inclusion & cohesion": {
                        "Gender equality": {},
                        "Anti-discrimination": {},
                        "Civic inclusion": {}
                    }
                },
                "Economic & Produced Capital": {
                    "Infrastructure & innovation": {
                        "Fixed capital": {},
                        "Infrastructure quality": {},
                        "Innovation capacity": {}
                    },
                    "Wealth sustainability": {
                        "Adjusted savings": {},
                        "Resource depletion": {}
                    }
                }
            }
        },
        # "Ageing-friendly": {
        #     "Outdoor Environment & Mobility": {
        #         "Physical environment": {
        #             "Walkability": {
        #                 "Pavement & sidewalks": {},
        #                 "Street crossings": {}
        #             },
        #             "Accessibility of public spaces": {
        #                 "Parks & green spaces": {},
        #                 "Seating & rest areas": {}
        #             }
        #         },
        #         "Transportation": {
        #             "Public transport": {
        #                 "Affordability": {},
        #                 "Reliability": {}
        #             },
        #             "Mobility services": {
        #                 "Paratransit": {},
        #                 "Community transport": {}
        #             }
        #         }
        #     },
        #     "Housing & Living Environment": {
        #         "Housing design": {
        #             "Accessibility": {
        #                 "Step-free access": {},
        #                 "Adaptable interior": {}
        #             },
        #             "Safety & comfort": {
        #                 "Thermal comfort": {},
        #                 "Safety devices": {}
        #             }
        #         },
        #         "Affordability & availability": {
        #             "Affordability": {},
        #             "Availability": {}
        #         }
        #     },
        #     "Social Participation": {
        #         "Cultural & recreational opportunities": {
        #             "Venue access": {},
        #             "Activity diversity": {}
        #         },
        #         "Community participation": {
        #             "Intergenerational": {},
        #             "Inclusive participation": {}
        #         }
        #     },
        #     "Respect & Social Inclusion": {
        #         "Attitudes towards older people": {
        #             "Societal attitudes": {},
        #             "Representation": {}
        #         },
        #         "Interpersonal relations": {
        #             "Family relations": {},
        #             "Community relations": {}
        #         }
        #     },
        #     "Civic Participation & Employment": {
        #         "Employment opportunities": {
        #             "Work flexibility": {},
        #             "Learning & skills": {}
        #         },
        #         "Civic engagement": {
        #             "Volunteering": {},
        #             "Political participation": {}
        #         }
        #     },
        #     "Communication & Information": {
        #         "Communication channels": {
        #             "Traditional media": {},
        #             "Digital inclusion": {}
        #         },
        #         "Information delivery": {
        #             "Readability": {},
        #             "Availability": {}
        #         }
        #     },
        #     "Community Support & Health Services": {
        #         "Community support": {
        #             "Social care": {},
        #             "Home help": {}
        #         },
        #         "Health services": {
        #             "Primary care": {},
        #             "Long-term & palliative care": {
        #                 "Long-term care": {},
        #                 "Palliative care": {}
        #             }
        #         }
        #     },
        #     "Security & Safety": {
        #         "Personal safety": {
        #             "Crime prevention": {},
        #             "Emergency response": {}
        #         },
        #         "Financial security": {
        #             "Income security": {},
        #             "Consumer protection": {}
        #         }
        #     }
        # }
}


# Ordered patterns from more specific to less specific; ISO-like first.
TEMP_NAME_PATTERNS = [
    ("quarter", re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?Q(?P<q>[1-4])\b", re.I)),
    ("quarter", re.compile(r"\bQ(?P<q>[1-4])[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    ("quarter", re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?T(?P<q>[1-4])\b", re.I)),
    ("quarter", re.compile(r"\bT(?P<q>[1-4])[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    ("semester", re.compile(r"\bS(?P<s>[12])[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    ("semester", re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?S(?P<s>[12])\b", re.I)),
    ("week",    re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?W(?P<w>[0-5]\d)\b", re.I)),
    ("week",    re.compile(r"\bW(?P<w>[0-5]\d)[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    # YYYY-MM / YYYY_M / YYYYMM
    ("month",   re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_/ ]?(?P<m>0?[1-9]|1[0-2])\b")),
    ("month",   re.compile(r"\b(?P<y>(18|19|20)\d{2})(?P<m>0[1-9]|1[0-2])\b")),
    # YYYY-MM-DD / YYYYMMDD
    ("date",     re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-/_ ]?(?P<m>0[1-9]|1[0-2])[\-/_ ]?(?P<d>0[1-9]|[12]\d|3[01])\b")),
    # Year alone
    ("year",    re.compile(r"\b(?P<y>(18|19|20)\d{2})\b")),
]
