from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table, Enum, Date, JSON, DateTime, Float, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy_utils import EmailType, UUIDType

# SQLAlchemy model 생성 전에 반드시 import
from .database import Base

# 2단계 association
membership = Table('membership', Base.metadata,
                          Column('user_id', Integer, ForeignKey("users.user_id"), primary_key=True),
                          Column('project_id', Integer, ForeignKey("projects.project_id"), primary_key=True),
                          Column('role', Enum("Manager","Member"))
                          )
# 1단계 association
# class Membership(Base):
#     __tablename__ = "membership"
#     user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
#     project_id = Column(Integer, ForeignKey("projets.project_id", primary_key=True))
#     role = Column(Enum("Manager", "Member"))
#     user = relationship("User", back_populates="membership")
#     project = relationship("Project", back_populates="membership")


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    # role = Column()
    organization = Column(Integer, ForeignKey("organizations.organization_id"))
    username = Column(String, unique=True)
    email = Column(EmailType, unique=True)      # admin@organization.~~~
    phonenumber = Column(String, default='')
    realname = Column(String, default='')       # UTF-8
    is_initiated = Column(Boolean, default=False)    # invitation pending or not
    department = Column(String, default='')
    position = Column(String, default='')

    # 1단계 association
    # projects = relationship("Membership", back_populates='users')
    projects = relationship("Project", secondary=membership, back_populates='users')

class ProjectType(Base):
    __tablename__ = "project_types"
    project_type_id = Column(Integer, primary_key=True, index=True)
    category = Column(Enum("HR", "Univ"))  # HR / Univ (Scope)
    project_type_name = Column(String)   # 상품 이름 (퇴사예측)


class Project(Base):
    __tablename__ = "projects"
    project_id = Column(Integer, primary_key=True, index=True)
    project_type = Column(Integer, ForeignKey("project_types.project_type_id"))
    organization = Column(Integer, ForeignKey("organizations.organization_id"))
    license = Column(Integer, ForeignKey("licenses.license_id"))
    last_main_order = Column(Integer, ForeignKey("main_orders.main_order_id"))

    # 1단계 association
    # users = relationship("Membership", back_populates='projects')
    users = relationship("User", secondary=membership, back_populates='projects')


class License(Base):
    __tablename__ = 'licences'
    license_id = Column(Integer, primary_key=True, index=True)
    code = Column(UUIDType(binary=False), unique=True)
    predict_period = Column(Enum("monthly", "yearly", "half", "quarter"))   # 예측 주기
    start_date = Column(Date)  # license 시작 날짜
    end_date = Column(Date)    # license 끝 날짜
    organization = Column(Integer, ForeignKey("organizations.organization_id"))
    project_type = Column(Integer, ForeignKey("project_types.project_type_id"))
    description = Column(String)
    is_demo = Column(Boolean, default=False)       # default false
    is_activated = Column(Boolean, default=False)  # default false


# Public S3, S3 bucket은 고정. 고정된 S3 bucket에 지정된 path만 적용되면 됨.
# s3://bucket_name/organization_id/filename.png
class Organization(Base):
    __tablename__ = "organizations"
    organization_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    url = Column(String)
    bi = Column("FILEFIELD", default="ALAB favicon")            # public s3 path, url
    admin_user = Column(Integer, ForeignKey("users.user_id"), nullable=True)


class Step(Base):
    __tablename__ = "steps"
    step_id = Column(Integer, primary_key=True, index=True)
    step_number = Column(Integer)
    title = Column(String)
    icon = Column(Integer)      # public s3 path?
    status = Column(Boolean, default=False)
    content_title = Column(String)
    content_desc1 = Column(String, nullable=True)
    content_desc2 = Column(String, nullable=True)
    project_type = Column(Integer, ForeignKey("project_types.project_type_id"))


class StepResponse(Base):
    __tablename__ = "step_responses"
    step_response_id = Column(Integer, primary_key=True, index=True)
    project = Column(Integer, ForeignKey("projects.project_id"))

    response = Column(JSON)     # 현재는 present / future 2개, 추후 변경 가능


class MainOrder(Base):
    __tablename__ = "main_orders"
    main_order_id = Column(Integer, primary_key=True, index=True)
    project = Column(Integer, ForeignKey("projects.project_id"))
    step_response


    # T, S, I Order는 relationship으로 처리




####################################################

# in optimizer
membership = Table('membership', Base.metadata,
                          Column('researcher_id', Integer, ForeignKey("researchers.researcher_id"), primary_key=True),
                          Column('experiment_id', Integer, ForeignKey("experiments.experiment_id"), primary_key=True),
                          )

class Researcher(Base):
    __tablename__ = "researchers"
    researcher_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    researcher_sub = Column(String, unique=True, index=True)        # keycloak "sub"

    experiments = relationship("Experiment", secondary=membership, back_populates='researchers')


# optimizer project 와 다른 service project mapping
class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_name = Column(String)        # 외부 project_id (ex. lean_001)
    users = Column(Integer, ForeignKey("users.user_id"))      # AFK
    tag = Column(String)        # HR / Univ / Lean (Scope)

    researchers = relationship("Project", secondary=membership, back_populates='experiments')


# Data App
class UploadedDataset(Base):
    __tablename__= "uploaded_datasets"
    uploaded_dataset_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # 기존의 project
    experiment = Column(Integer, ForeignKey("experiments.experiment_id"), cascade="all")
    # cascade all / delete / delete_orphan
    file_path = Column(String)
    file_size = Column(String)
    encoding = Column(String)           # UTF-8, CP949 ...
    label = Column(String)
    dataset_type = Column(String)       # training set, validation set, inference set ...
    created_at = Column(DateTime)       # Data Version
    updated_at = Column(DateTime)


class DatasetMetadata(Base):
    __tablename__= "dataset_metadatas"
    dataset_metadata_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    row_count = Column(Integer)     # data record 수
    missing_rate = Column(JSON)     # 각 column 별 결측치 비율
    data_formats = Column(JSON)     # 각 column 별 datatype 및 format


class Train(Base):
    __tablename__ = "trains"
    train_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment = Column(Integer, ForeignKey("experiments.experiment_id"))
    # models.ForeignKey(ProjectTable, on_delete=models.PROTECT, related_name='project_train')
    dag_run_id = Column(String)
    datafile = Column(Integer, ForeignKey("uploaded_datasets.upload_dataset_id"))   # AFK
    # models.ForeignKey(UploadedDataset, on_delete=models.PROTECT, related_name='uploaded_file')
    label = Column(String)
    type = Column(Integer, ForeignKey("model_types.model_type_id"))
    # models.ForeignKey(ModelType, models.CASCADE, related_name='train_type', blank=True, null=True)
    tag = Column(String)  # AutoML 인지, SKLearn( = Manual 로 하나만...) 인지.
    ratio = Column(Float, default=0.3)
    sampling = Column(Enum("STD", "OVER", "UNDER"), default="STD")
    coverage = Column(Enum("FULL", "HALF", "DEMO"), default="FULL")
    # is_finished = models.BooleanField(null=True, default=None)
    is_finished = Column(Boolean, default=False)
    is_failed = Column(Boolean, default=False)


class SplitDataset(Base):
    __tablename__= "split_datasets"
    split_dataset_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    train = Column(Integer, ForeignKey("trains.train_id"))
    train_set = Column(String)          # s3 file path
    # models.FileField(upload_to=datetime.datetime.today().strftime("%Y-%m-%d") + '/train_set/', blank=True, null=True)
    # valid_set = models.FileField(upload_to=datetime.datetime.today().strftime("%Y-%m-%d") + '/validation_set/', blank=True, null=True)
    test_set = Column(String)


class TrainEncoder(Base):
    __tablename__= "train_encoders"
    train_encoder_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    train = Column(Integer, ForeignKey("trains.train_id"))
    train_encoder_pkl = Column(String)          # s3 encoder.pkl path


class TrainScaler(Base):
    __tablename__= "train_scalers"
    train_scaler_id = "train_scalers"
    train = Column(Integer, ForeignKey("trains.train_id"))
    train_scaler_pkl = Column(String)           # s3 scaler.pkl path


class ModelType(Base):
    __tablename__= "model_types"
    model_type_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_type = Column(String)     # Regression / Classification / Recommendation ...
    autoML = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=True))
    sklearn = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=True))
    metrics = Column(ARRAY(String, dimensions=1, zero_indexes=True, as_tuple=True))


class ModelService(Base):
    __tablename__ = "model_services"
    model_service_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    url = Column(String)        # model dns
    is_available = Column(Boolean, default=False)       # 서빙 상태인지 아닌지 (inference 가능한지)
    is_locked = Column(Boolean, default=False)          # semaphore


class Model(Base):
    __tablename__= "models"
    model_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    algorithm = Column(String)
    version = Column(Integer, default=1)        # model version from MLflow
    train = Column(Integer, ForeignKey("trains.train_id"))
    service = Column(Integer, ForeignKey("model_services.model_service_id"))
    # encoder = Column(Integer, ForeignKey("train_encoders.train_encoder_id"))
    # scaler = Column(Integer, ForeignKey("train_scaler.model_scaler_id"))
    # encoder, scaler 모두 train 통해서 접근 가능
    description = Column(String, default="")


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    model_metric_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model = Column(Integer, ForeignKey("models.model_id"))
    feature_list = Column(JSON, nullable=True)
    metrics = Column(JSON)          # MAE, MAPE, accuracy, etc...
    feature_importance = Column(JSON, nullable=True)
    bias = Column(Float, nullable=True)
    default_values = Column(JSON, nullable=True)
    params = Column(JSON, nullable=True)



