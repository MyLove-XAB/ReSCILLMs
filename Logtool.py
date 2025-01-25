import logging


def CreateLog(name, filename, level=logging.DEBUG, sh_level=logging.DEBUG, fh_level=logging.DEBUG,
              t_stamp=True, add_sh=True, add_fh=False):
    """
    Create Log to record some running information

    :param name:  日志收集器名字
    :param level: 日志收集器的等级
    :param filename:  日志文件的名称
    :param sh_level:  控制台输出日志的等级
    :param fh_level:    文件输出日志的等级
    :param t_stamp: 时间戳格式
    :param add_sh: 增加steamHandler
    :param add_fh: 增加fileHandler
    :return: 返回创建好的日志收集器
    """

    # 1、创建日志收集器
    log = logging.getLogger(name)

    # 2、创建日志收集器的等级
    log.setLevel(level=level)

    # 4、设置日志的输出格式
    if t_stamp:
        formats = "%(created)f - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s"  # 时间戳格式
    else:
        formats = "%(asctime)s - [%(funcName)s-->line:%(lineno)d] - %(levelname)s:%(message)s"  # 刻度时间格式
    log_format = logging.Formatter(fmt=formats)

    # 3、创建日志收集渠道和等级
    if add_fh:
        fh = logging.FileHandler(filename=filename, encoding="utf-8")
        # fh1 = handlers.TimedRotatingFileHandler(filename=filename,when="D",interval=1,backupCount=10,encoding="utf-8")
        fh.setLevel(level=fh_level)
        log.addHandler(fh)
        fh.setFormatter(log_format)
    if add_sh:
        sh = logging.StreamHandler()
        sh.setLevel(level=sh_level)
        log.addHandler(sh)
        sh.setFormatter(log_format)

    return log


if __name__ == '__main__':
    log = CreateLog(name="rose_log", level=logging.DEBUG, filename="test_log.log", sh_level=logging.DEBUG,
                    fh_level=logging.DEBUG, t_stamp=False, add_fh=True)
    log.info(msg="--------debug--------")
    log.info(msg="--------info--------")
    log.info(msg="--------warning--------")
    log.info(msg="--------error--------")
    log.info(msg="--------critical--------")